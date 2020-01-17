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

static unsigned width, height, width_in_mb, height_in_mb;
static unsigned img_id = 0;

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
  stream << fixed << setprecision(1) << num;
  return stream.str();
}

static double prev_residue_mean = 0;
static double prev_pixel_mse = 0;

void smooth_mv(vector<vector<pair<int, int>>> & mv)
{
  (void) mv;
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
  Mat curr_mat = convert_yuv_to_bgr(raster);
  Mat mv_mat = curr_mat.clone();
  Mat residue_mat = curr_mat.clone();

  vector<vector<pair<int, int>>> mv(height_in_mb, vector<pair<int, int>>(width_in_mb));
  vector<vector<uint32_t>> residue(height_in_mb, vector<uint32_t>(width_in_mb));

  /* record motion vectors and residues */
  frame.macroblocks().forall_ij(
    [&](MacroblockType & macroblock, unsigned int mb_col, unsigned int mb_row)
    {
      /* skip key frame for now */
      if (not macroblock.inter_coded()) {
        return;
      }

      if (macroblock.header().reference() != LAST_FRAME) {
        throw runtime_error("Only supports visualizing reference frame == LAST_FRAME");
      }

      /* record motion vector */
      const auto & curr_mv = macroblock.base_motion_vector();
      mv[mb_row][mb_col] = {curr_mv.x() >> 3, curr_mv.y() >> 3};

      /* record residue */
      auto temp_mb = temp_raster.macroblock(mb_col, mb_row);
      TwoDSubRange<uint8_t, 16, 16> & prediction = temp_mb.Y.mutable_contents();

      const auto & original_mb = raster.macroblock(mb_col, mb_row);
      original_mb.Y().inter_predict(curr_mv, reference.Y(), prediction);

      residue[mb_row][mb_col] = Encoder::sse(original_mb.Y(), prediction);
    }
  );

  /* smooth motion vectors */
  smooth_mv(mv);

  /* draw motion vectors */
  for (unsigned r = 0; r < height_in_mb; r++) {
    for (unsigned c = 0; c < width_in_mb; c++) {
      int curr_row = r * 16;
      int curr_col = c * 16;
      int prev_row = curr_row + mv[r][c].second;
      int prev_col = curr_col + mv[r][c].first;

      arrowedLine(mv_mat, Point(prev_col, prev_row), Point(curr_col, curr_row),
                  Scalar(255, 255, 255));
    }
  }
  imwrite(to_string(img_id) + "-mv.jpg", mv_mat);

  /* average residues and draw */
  uint64_t frame_residue_sum = 0;
  unsigned stride = 8;
  for (unsigned r = 0; r + stride < height_in_mb; r += stride) {
    for (unsigned c = 0; c + stride < width_in_mb; c += stride) {
      uint64_t residue_sum = 0;
      for (unsigned i = r; i < r + stride; i++) {
        for (unsigned j = c; j < c + stride; j++) {
          residue_sum += residue[i][j];
        }
      }
      frame_residue_sum += residue_sum;

      double residue_mean = (double) residue_sum / (16 * 16 * stride * stride);
      putText(residue_mat, fixed_precision(residue_mean),
              Point(c * 16 + 10, r * 16 + 20), FONT_HERSHEY_COMPLEX_SMALL,
              0.7, Scalar(0, 255, 0));
    }
  }

  double frame_residue_mean = (double) frame_residue_sum / (width * height);
  string text = "Avg residue: " + fixed_precision(frame_residue_mean) + " ("
                + fixed_precision(frame_residue_mean - prev_residue_mean) + ")";
  putText(residue_mat, text, Point(10, 50), FONT_HERSHEY_COMPLEX_SMALL,
          1.5, Scalar(255, 255, 255));

  /* calculate pixel-wise MSE and draw */
  double frame_pixel_mse = 0;
  if (not frame.header().key_frame()) {
    frame_pixel_mse = mse(curr_mat, convert_yuv_to_bgr(reference));
  }

  text = "Pixel MSE: " + fixed_precision(frame_pixel_mse) + " ("
         + fixed_precision(frame_pixel_mse - prev_pixel_mse) + ")";
  putText(residue_mat, text, Point(10, 80), FONT_HERSHEY_COMPLEX_SMALL,
          1.5, Scalar(255, 255, 255));

  img_id++;
  imwrite(to_string(img_id) + "-residue.jpg", residue_mat);

  prev_residue_mean = frame_residue_mean;
  prev_pixel_mse = frame_pixel_mse;
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
