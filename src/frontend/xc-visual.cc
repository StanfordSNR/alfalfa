/* -*-mode:c++; tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2 -*- */

/* Copyright 2013-2018 the Alfalfa authors
                       and the Massachusetts Institute of Technology

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:

      1. Redistributions of source code must retain the above copyright
         notice, this list of conditions and the following disclaimer.

      2. Redistributions in binary form must reproduce the above copyright
         notice, this list of conditions and the following disclaimer in the
         documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. */

#include <getopt.h>
#include <climits>
#include <iostream>
#include <string>
#include <array>
#include <optional>
#include <sstream>
#include <iomanip>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include "frame.hh"
#include "decoder.hh"
#include "ivf.hh"
#include "encoder.hh"

using namespace std;
using namespace cv;

unsigned width, height, width_in_mb, height_in_mb;

void usage_error(const string & program_name)
{
  cerr <<
  "Usage: " << program_name << " <input>"
  << endl;
}

Mat convert_yuv_to_bgr(const VP8Raster & raster)
{
  /* create a YUV Mat from raster */
  const auto & Y = raster.Y();
  const auto & U = raster.U();
  const auto & V = raster.V();

  vector<char> buf_src(Y.height() * Y.width()
                       + U.height() * U.width()
                       + V.height() * V.width());
  unsigned i = 0;

  for (unsigned r = 0; r < Y.height(); r++) {
    for (unsigned c = 0; c < Y.width(); c++) {
      buf_src[i++] = Y.at(c, r);
    }
  }

  for (unsigned r = 0; r < U.height(); r++) {
    for (unsigned c = 0; c < U.width(); c++) {
      buf_src[i++] = U.at(c, r);
    }
  }

  for (unsigned r = 0; r < V.height(); r++) {
    for (unsigned c = 0; c < V.width(); c++) {
      buf_src[i++] = V.at(c, r);
    }
  }

  Mat yuv(Y.height() + U.height(), Y.width(), CV_8UC1, buf_src.data());

  /* convert YUV to BGR */
  Mat bgr;
  cvtColor(yuv, bgr, COLOR_YUV2BGR_I420);

  return bgr;
}

double mse(const Mat & m1, const Mat & m2)
{
  Mat m = m1 - m2;
  m = m.mul(m);
  Scalar s = sum(m);
  double sse = s.val[0] + s.val[1] + s.val[2];
  return sse / (m1.channels() * m1.total());
}

string fixed_precision(double num)
{
  std::stringstream stream;
  stream << fixed << setprecision(2) << num;
  return stream.str();
}

template<class FrameHeaderType, class MacroblockType>
void decode_and_visualize(Decoder & decoder,
                          const Frame<FrameHeaderType, MacroblockType> & frame)
{
  /* save the reference raster before decode_frame */
  const VP8Raster & reference = decoder.get_references().at(LAST_FRAME);

  /* decode the current raster */
  auto output = decoder.decode_frame(frame);
  if (not output.first) {
    return;
  }
  const VP8Raster & raster = output.second;
  VP8Raster temp_raster(width, height);

  /* convert the current raster to OpenCV Mat (YUV to BGR) */
  Mat bgr = convert_yuv_to_bgr(raster);

  vector<vector<uint32_t>> residue(height_in_mb, vector<uint32_t>(width_in_mb));

  /* overlay frame stats (motion vectors, residues) on Mat */
  frame.macroblocks().forall_ij(
    [&](MacroblockType & macroblock, unsigned int mb_col, unsigned int mb_row)
    {
      int curr_col = mb_col * 16;
      int curr_row = mb_row * 16;

      if (not macroblock.inter_coded()) {
        circle(bgr, Point(curr_col, curr_row), 1, Scalar(0, 0, 0));
        return;
      }

      if (macroblock.header().reference() != LAST_FRAME) {
        throw runtime_error("Only supports visualizing reference frame == LAST_FRAME");
      }

      /* draw motion vector */
      const auto & mv = macroblock.base_motion_vector();

      int prev_col = curr_col + (mv.x() >> 3);
      int prev_row = curr_row + (mv.y() >> 3);

      arrowedLine(bgr, Point(prev_col, prev_row), Point(curr_col, curr_row),
                  Scalar(255, 255, 255));

      /* record residue */
      auto temp_mb = temp_raster.macroblock(mb_col, mb_row);
      TwoDSubRange<uint8_t, 16, 16> & prediction = temp_mb.Y.mutable_contents();

      const auto & original_mb = raster.macroblock(mb_col, mb_row);
      original_mb.Y().inter_predict(mv, reference.Y(), prediction);

      residue[mb_row][mb_col] = Encoder::sse(original_mb.Y(), prediction);
    }
  );

  unsigned stride = 8;
  for (unsigned r = 0; r + stride < height_in_mb; r += stride) {
    for (unsigned c = 0; c + stride < width_in_mb; c += stride) {
      uint64_t residue_sum = 0;
      for (unsigned i = r; i < r + stride; i++) {
        for (unsigned j = c; j < c + stride; j++) {
          residue_sum += residue[i][j];
        }
      }
      double residue_mean = (double) residue_sum / (16 * 16 * stride * stride);

      putText(bgr, fixed_precision(residue_mean),
              Point(c * 16 + 16, r * 16 + 16), FONT_HERSHEY_COMPLEX_SMALL,
              1.0, Scalar(0, 255, 0));
    }
  }

  /* display Mat */
  imshow("xc-visual", bgr);
  waitKey(0);
}

int main(int argc, char * argv[])
{
  if (argc < 1) {
    abort();
  }

  if (argc != 2) {
    usage_error(argv[0]);
    return EXIT_FAILURE;
  }

  string input_video = argv[1];
  const IVF ivf(input_video);

  width = ivf.width();
  height = ivf.height();

  width_in_mb = VP8Raster::macroblock_dimension(width);
  height_in_mb = VP8Raster::macroblock_dimension(height);

  Decoder decoder(width, height);

  for (unsigned frame_id = 0; frame_id < ivf.frame_count(); frame_id++) {
    UncompressedChunk decompressed_frame = decoder.decompress_frame(ivf.frame(frame_id));

    if (decompressed_frame.key_frame()) {
      KeyFrame frame = decoder.parse_frame<KeyFrame>(decompressed_frame);
      decode_and_visualize(decoder, frame);
    } else {
      InterFrame frame = decoder.parse_frame<InterFrame>(decompressed_frame);
      decode_and_visualize(decoder, frame);
    }
  }

  return EXIT_SUCCESS;
}
