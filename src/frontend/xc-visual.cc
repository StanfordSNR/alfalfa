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

using namespace std;
using namespace cv;

unsigned width, height, width_in_mb, height_in_mb;

void usage_error(const string & program_name)
{
  cerr <<
  "Usage: " << program_name << " <input> <output>"
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

template<class FrameType>
void write_frame(const VideoWriter & /* video_writer */,
                 const VP8Raster & raster,
                 const FrameType & /* frame */)
{
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

  Mat bgr;
  cvtColor(yuv, bgr, COLOR_YUV2BGR_I420);

  imshow("BGR", bgr);
  waitKey(0);
}

int main(int argc, char * argv[])
{
  if (argc < 1) {
    abort();
  }

  if (argc != 3) {
    usage_error(argv[0]);
    return EXIT_FAILURE;
  }

  string input_video = argv[1];
  const IVF ivf(input_video);

  width = ivf.width();
  height = ivf.height();

  width_in_mb = VP8Raster::macroblock_dimension(width);
  height_in_mb = VP8Raster::macroblock_dimension(height);

  /* create a video writer for the output video */
  string output_video = argv[2];
  VideoWriter video_writer(output_video, VideoWriter::fourcc('M', 'J', 'P', 'G'),
                           1, Size(width, height), true);

  Decoder decoder(width, height);

  for (unsigned frame_id = 0; frame_id < ivf.frame_count(); frame_id++) {
    UncompressedChunk decompressed_frame = decoder.decompress_frame(ivf.frame(frame_id));

    if (decompressed_frame.key_frame()) {
      KeyFrame frame = decoder.parse_frame<KeyFrame>(decompressed_frame);

      auto output = decoder.decode_frame(frame);
      if (output.first) {
        write_frame(video_writer, output.second.get(), frame);
      }
    } else {
      InterFrame frame = decoder.parse_frame<InterFrame>(decompressed_frame);

      auto output = decoder.decode_frame(frame);
      if (output.first) {
        write_frame(video_writer, output.second.get(), frame);
      }
    }
  }

  return EXIT_SUCCESS;
}
