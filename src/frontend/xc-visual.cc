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

#include "frame.hh"
#include "decoder.hh"
#include "ivf.hh"
#include "display.hh"

using namespace std;

unsigned width, height, width_in_mb, height_in_mb;

void usage_error(const string & program_name)
{
  cerr <<
  "Usage: " << program_name << " [options] <ivf>\n\n"
  "Options:\n"
  "-f, --frame <arg>              Print information for frame #<arg>"
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
void display_mb(VideoDisplay & display,
                VP8Raster & raster,
                FrameType & frame,
                const unsigned mb_col,
                const unsigned mb_row)
{
  for (unsigned c = 0; c < width_in_mb; c++) {
    for (unsigned r = 0; r < height_in_mb; r++) {
      if (not (c / 4 == mb_col and r / 4 == mb_row)) {
        raster.macroblock(c, r).white_out();
      }
    }
  }

  frame.macroblocks().forall_ij(
    [&](auto & mb, unsigned int c, unsigned int r)
    {
      if (not (c / 4 == mb_col and r / 4 == mb_row)) {
        return;
      }

      cout << "Macroblock [" << c << ", " << r << "]" << endl;
      cout << "Prediction Mode: " << mbmode_names[mb.y_prediction_mode()] << endl;

      if (mb.inter_coded()) {
        // cout << "Base Motion Vector: (" << mb.base_motion_vector().x()
        //      << ", " << mb.base_motion_vector().y() << ")" << endl;
        // cout << "Reference: " << reference_frame_names[mb.header().reference()] << endl;
      }
      cout << endl;
    });

  display.draw(raster);
  getchar();
}

template<class FrameType>
void display_frame(VideoDisplay & display,
                   const VP8Raster & raster,
                   const FrameType & frame)
{
  for (unsigned c = 0; c < width_in_mb / 4; c++) {
    for (unsigned r = 0; r < height_in_mb / 4; r++) {
      VP8Raster raster_copy(width, height);
      raster_copy.copy_from(raster);

      display_mb(display, raster_copy, frame, c, r);
    }
  }
}

int main(int argc, char * argv[])
{
  if (argc < 1) {
    abort();
  }

  const option cmd_line_opts[] = {
    {"frame",  required_argument, nullptr, 'f'},
    { nullptr, 0,                 nullptr,  0 }
  };

  unsigned target_frame_number = UINT_MAX;

  while (true) {
    const int opt = getopt_long(argc, argv, "f:", cmd_line_opts, nullptr);
    if (opt == -1) {
      break;
    }

    switch (opt) {
    case 'f':
      target_frame_number = stoul(optarg);
      break;

    default:
      usage_error(argv[0]);
      return EXIT_FAILURE;
    }
  }

  if (optind != argc - 1) {
    usage_error(argv[0]);
    return EXIT_FAILURE;
  }

  string video_file = argv[optind];

  const IVF ivf(video_file);

  width = ivf.width();
  height = ivf.height();

  width_in_mb = VP8Raster::macroblock_dimension(width);
  height_in_mb = VP8Raster::macroblock_dimension(height);

  Decoder decoder(width, height);
  VideoDisplay display(decoder.example_raster());

  for (unsigned frame_id = 0; frame_id < ivf.frame_count(); frame_id++) {
    if (frame_id > target_frame_number) {
      break;
    }

    UncompressedChunk decompressed_frame = decoder.decompress_frame(ivf.frame(frame_id));

    if (decompressed_frame.key_frame()) {
      KeyFrame frame = decoder.parse_frame<KeyFrame>(decompressed_frame);
      auto output = decoder.decode_frame(frame);
      if (frame_id == target_frame_number and output.first) {
        display_frame(display, output.second.get(), frame);
      }
    } else {
      InterFrame frame = decoder.parse_frame<InterFrame>(decompressed_frame);
      auto output = decoder.decode_frame(frame);
      if (frame_id == target_frame_number and output.first) {
        display_frame(display, output.second.get(), frame);
      }
    }
  }

  return EXIT_SUCCESS;
}
