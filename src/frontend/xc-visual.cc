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

static array<string, 10> mbmode_names =
  { "DC_PRED", "V_PRED", "H_PRED", "TM_PRED", "B_PRED",
    "NEARESTMV", "NEARMV", "ZEROMV", "NEWMV", "SPLITMV" };

static array<string, 14> bmode_names =
  { "B_DC_PRED", "B_TM_PRED", "B_VE_PRED", "B_HE_PRED", "B_LD_PRED",
    "B_RD_PRED", "B_VR_PRED", "B_VL_PRED", "B_HD_PRED", "B_HU_PRED",
    "LEFT4X4", "ABOVE4X4", "ZERO4X4", "NEW4X4" };

static array<string, 4> reference_frame_names =
  { "CURRENT_FRAME", "LAST_FRAME", "GOLDEN_FRAME", "ALTREF_FRAME" };

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
  VP8Raster temp_raster(raster.width(), raster.height());

  /* convert the current raster to OpenCV Mat (YUV to BGR) */
  Mat bgr = convert_yuv_to_bgr(raster);

  uint64_t residue_sum = 0;
  int64_t mv_x_sum = 0, mv_y_sum = 0;

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

      mv_x_sum += mv.x();
      mv_y_sum += mv.y();

      int prev_col = curr_col + (mv.x() >> 3);
      int prev_row = curr_row + (mv.y() >> 3);

      arrowedLine(bgr, Point(prev_col, prev_row), Point(curr_col, curr_row),
                  Scalar(255, 255, 255));

      /* draw residue */
      auto temp_mb = temp_raster.macroblock(mb_col, mb_row);
      TwoDSubRange<uint8_t, 16, 16> & prediction = temp_mb.Y.mutable_contents();

      const auto & original_mb = raster.macroblock(mb_col, mb_row);
      original_mb.Y().inter_predict(mv, reference.Y(), prediction);

      residue_sum += Encoder::sse(original_mb.Y(), prediction);
    }
  );

  cerr << "Sum of motion vectors = " << mv_x_sum << " " << mv_y_sum << endl;
  cerr << "Sum of residues = " << residue_sum << endl;

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
