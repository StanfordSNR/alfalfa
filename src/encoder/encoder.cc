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

#include <algorithm>
#include <array>
#include <cstdio>
#include <limits>
#include <utility>
#include <chrono>

#include "block.hh"
#include "encoder.hh"
#include "frame_header.hh"
#include "tokens.hh"
#include "tokenize.hh"

using namespace std;

#include "encode_inter.cc"
#include "encode_intra.cc"
#include "reencode.cc"
#include "size_estimation.cc"

unsigned Encoder::calc_prob( unsigned false_count, unsigned total )
{
  if ( false_count == 0 ) {
    return 0;
  } else {
    return max( 1u, min( 255u, 256 * false_count / total ) );
  }
}

QuantIndices::QuantIndices()
  : y_ac_qi(), y_dc(), y2_dc(), y2_ac(), uv_dc(), uv_ac()
{}

template <unsigned int size>
void VP8Raster::Block<size>::inter_predict( const MotionVector & mv,
                                            const TwoD<uint8_t> & reference,
                                            TwoD<uint8_t> & output ) const
{
  TwoDSubRange<uint8_t, size, size> subrange( output, 0, 0 );
  inter_predict( mv, reference, subrange );
}

/* Encoder */
Encoder::Encoder( const uint16_t s_width,
                  const uint16_t s_height,
                  const bool two_pass,
                  const EncoderQuality quality )
  : decoder_state_( s_width, s_height ),
    references_( width(), height() ),
    safe_references_( references_ ), has_state_( false ), costs_(),
    two_pass_encoder_( two_pass ), encode_quality_( quality )
{
  costs_.fill_mode_costs();
}

Encoder::Encoder( const Decoder & decoder, const bool two_pass,
                  const EncoderQuality quality )
  : decoder_state_( decoder.get_state() ), references_( decoder.get_references() ),
    safe_references_( references_ ), has_state_( true ), costs_(),
    two_pass_encoder_( two_pass ), encode_quality_( quality )
{
  costs_.fill_mode_costs();
}

Encoder::Encoder( const Encoder & encoder )
  : decoder_state_( encoder.decoder_state_ ),
    references_( encoder.references_ ),
    safe_references_( encoder.safe_references_ ),
    has_state_( encoder.has_state_ ), costs_( encoder.costs_ ),
    two_pass_encoder_( encoder.two_pass_encoder_ ),
    encode_quality_( encoder.encode_quality_ ),
    loop_filter_level_( encoder.loop_filter_level_ ),
    last_y_ac_qi_( encoder.last_y_ac_qi_ ),
    encode_stats_( encoder.encode_stats_ )
{}

Encoder::Encoder( Encoder && encoder )
  : decoder_state_( move( encoder.decoder_state_ ) ),
    references_( move( encoder.references_ ) ),
    safe_references_( move( encoder.safe_references_ ) ),
    has_state_( encoder.has_state_ ), costs_( move( encoder.costs_ ) ),
    two_pass_encoder_( encoder.two_pass_encoder_ ),
    encode_quality_( encoder.encode_quality_ ),
    key_frame_( move( encoder.key_frame_ ) ),
    subsampled_key_frame_( move( encoder.subsampled_key_frame_ ) ),
    inter_frame_( move( encoder.inter_frame_ ) ),
    subsampled_inter_frame_( move( encoder.subsampled_inter_frame_ ) ),
    loop_filter_level_( move( encoder.loop_filter_level_ ) ),
    last_y_ac_qi_( move( encoder.last_y_ac_qi_ ) ),
    encode_stats_( move( encoder.encode_stats_ ) )
{}

Encoder & Encoder::operator=( Encoder && encoder )
{
  decoder_state_ = move( encoder.decoder_state_ );
  references_ = move( encoder.references_ );
  safe_references_ = move( encoder.safe_references_ );
  has_state_ = encoder.has_state_;
  costs_ = move( encoder.costs_ );
  two_pass_encoder_ = encoder.two_pass_encoder_;
  encode_quality_ = encoder.encode_quality_;
  key_frame_ = move( encoder.key_frame_ );
  subsampled_key_frame_ = move( encoder.subsampled_key_frame_ );
  inter_frame_ = move( encoder.inter_frame_ );
  subsampled_inter_frame_ = move( encoder.subsampled_inter_frame_ );
  loop_filter_level_ = move( encoder.loop_filter_level_ );
  last_y_ac_qi_ = move( encoder.last_y_ac_qi_ );
  encode_stats_ = move( encoder.encode_stats_ );

  return *this;
}

uint32_t Encoder::minihash() const
{
  return static_cast<uint32_t>( DecoderHash( decoder_state_.hash(), references_.last.hash(),
                                references_.golden.hash(), references_.alternative.hash() ).hash() );
}

template<class FrameType>
vector<uint8_t> Encoder::write_frame( const FrameType & frame,
                                      const ProbabilityTables & prob_tables )
{
  // update the state
  update_decoder_state( frame );

  // update the references
  MutableRasterHandle raster { width(), height() };
  frame.decode( decoder_state_.segmentation, references_, raster );
  frame.loopfilter( decoder_state_.segmentation, decoder_state_.filter_adjustments, raster );
  RasterHandle immutable_raster( move( raster ) );
  frame.copy_to( immutable_raster, references_ );

  safe_references_.last = move( SafeReferences::load( references_.last ) );
  safe_references_.golden = move( SafeReferences::load( references_.golden ) );
  safe_references_.alternative = move( SafeReferences::load( references_.alternative ) );

  if ( encode_quality_ == REALTIME_QUALITY or encode_quality_ == CV_QUALITY ) {
    loop_filter_level_.reset( frame.header().loop_filter_level );
    last_y_ac_qi_.reset( frame.header().quant_indices.y_ac_qi );
  }

  return frame.serialize( prob_tables );
}

template<class FrameType>
vector<uint8_t> Encoder::write_frame( const FrameType & frame )
{
  return write_frame( frame, decoder_state_.probability_tables );
}

void Encoder::update_rd_multipliers( const Quantizer & quantizer )
{
  /* This is how VP8 sets the coefficients for rd-cost.
   * libvpx:vp8/encoder/rdopt.c:270
   */
  double q_ac = ( quantizer.y_ac < 160 ) ? quantizer.y_ac : 160.0 ;
  RATE_MULTIPLIER = q_ac * q_ac * 2.80;

  if ( RATE_MULTIPLIER > 1000 ) {
    DISTORTION_MULTIPLIER = 1;
    RATE_MULTIPLIER /= 100;
  }
  else {
    DISTORTION_MULTIPLIER = 100;
  }
}

/*
 * Based on libvpx:vp8/encoder/encodemb.c:413
 */
void Encoder::check_reset_y2( Y2Block & y2, const Quantizer & quantizer ) const
{
  int sum = 0;

  if ( quantizer.y2_dc >= 35 and quantizer.y2_ac >= 35 ) {
    return;
  }

  for ( size_t i = 0; i < 16; i++ ) {
    sum += abs( y2.coefficients().at( i ) );

    if ( sum >= 35 ) {
      return;
    }
  }

  // sum < 35
  for ( int i = 0; i < 16; i++ ) {
    y2.mutable_coefficients().at( i ) = 0;
  }
}

template<class FrameSubblockType>
void Encoder::trellis_quantize( FrameSubblockType & frame_sb,
                                const Quantizer & quantizer ) const
{
  struct TrellisNode
  {
    uint32_t rate;
    uint32_t distortion;
    uint32_t cost;
    int16_t coeff;

    uint8_t token;
    uint8_t next;
  };

  uint16_t dc_factor;
  uint16_t ac_factor;

  // select quantization factors based on macroblock type
  switch( frame_sb.type() ) {
  case BlockType::UV:
    dc_factor = quantizer.uv_dc;
    ac_factor = quantizer.uv_ac;
    break;

  case BlockType::Y2:
    dc_factor = quantizer.y2_dc;
    ac_factor = quantizer.y2_ac;
    break;

  default:
    dc_factor = quantizer.y_dc;
    ac_factor = quantizer.y_ac;
  }

  const size_t first_index = ( frame_sb.type() == BlockType::Y_after_Y2 ) ? 1 : 0;
  uint8_t coded_length = 0;

  /* how many tokens are we going to encode? */
  for ( size_t index = first_index; index < 16; index++ ) {
    if ( frame_sb.coefficients().at( zigzag.at( index ) ) ) {
      coded_length = index + 1;
    }
  }

  if ( coded_length == 0 ) {
    // everything is zero. our work is done here.
    for ( size_t i = 0; i < 16; i++ ) {
      frame_sb.mutable_coefficients().at( i ) = 0;
    }

    return;
  }

  const uint8_t LEVELS = 2;
  SafeArray<SafeArray<TrellisNode, LEVELS>, 17> trellis;

  // setting up the sentinel node for the trellis
  for ( size_t i = 0; i < LEVELS; i++ ) {
    TrellisNode & sentinel_node = trellis.at( coded_length ).at( i );
    sentinel_node.rate = 0;
    sentinel_node.distortion = 0;
    sentinel_node.token = DCT_EOB_TOKEN;
    sentinel_node.coeff = 0;
    sentinel_node.next = numeric_limits<uint8_t>::max();
  }

  // building the trellis first
  for ( uint8_t idx = coded_length; idx-- > first_index; ) {
    const int16_t original_coeff = frame_sb.coefficients().at( zigzag.at( idx ) );
    const int16_t quantized_coeff = original_coeff /
                                    ( idx == 0 ? dc_factor : ac_factor );

    // evaluate two quantizer levels: {q, q - 1}
    // it's possible to explore more
    for ( size_t q_shift = 0; q_shift < LEVELS; q_shift++ ) {
      TrellisNode & current_node = trellis.at( idx ).at( q_shift );

      int16_t candidate_coeff = quantized_coeff;

      if ( candidate_coeff < 0 ) {
        candidate_coeff += q_shift;
        candidate_coeff = min( ( int16_t )0, candidate_coeff );
      }
      else if ( candidate_coeff > 0 or q_shift == 0 )  {
        candidate_coeff -= q_shift;
        candidate_coeff = max( ( int16_t )0, candidate_coeff );
      }
      else { // candidate_coeff == 0 and q_shift != 0
        current_node = trellis.at( idx ).at( q_shift - 1 );
        continue;
      }

      int16_t diff = ( original_coeff - candidate_coeff * ( idx == 0 ? dc_factor : ac_factor ) );
      uint32_t sse = diff * diff;

      current_node.coeff = candidate_coeff;
      current_node.token = Costs::token_for_coeff( candidate_coeff );

      array<uint32_t, LEVELS> distortions;
      array<uint32_t, LEVELS> rates;
      array<uint32_t, LEVELS> rd_costs;

      // fill the trellis for current coeff index and select the best
      uint8_t best_next = numeric_limits<uint8_t>::max();
      uint32_t best_cost = numeric_limits<uint32_t>::max();

      for ( size_t next = 0; next < LEVELS; next++ ) {
        const TrellisNode & next_node = trellis.at( idx + 1 ).at( next );

        distortions[ next ] = trellis.at( idx + 1 ).at( next ).distortion + sse;
              rates[ next ] = trellis.at( idx + 1 ).at( next ).rate;

        if ( idx < 15 ) {
          size_t next_band = coefficient_to_band.at( idx + 1 );
          size_t current_context = prev_token_class.at( current_node.token );

          // cost of the next token based on the *current* context
          rates[ next ] += costs_.token_costs.at( frame_sb.type() )
                                             .at( next_band )
                                             .at( current_context )
                                             .at( next_node.token );
        }

        rd_costs[ next ] = rdcost( rates[ next ], distortions[ next ],
                                   RATE_MULTIPLIER, DISTORTION_MULTIPLIER );

        if ( rd_costs[ next ] < best_cost ) {
          best_cost = rd_costs[ next ];
          best_next = next;
        }
      }

      if ( current_node.coeff != 0 or trellis.at( idx + 1 ).at( best_next ).token != DCT_EOB_TOKEN ) {
        current_node.rate = rates[ best_next ] + Costs::coeff_base_cost( current_node.coeff );
        current_node.distortion = distortions[ best_next ];
        current_node.cost = rd_costs[ best_next ];
        current_node.next = best_next;
      }
      else {
        // this token is zero and the next one is EOB, so let's move EOB back here
        current_node.coeff = 0;
        current_node.rate = 0;
        current_node.distortion = sse;
        current_node.cost = rdcost( 0, sse, RATE_MULTIPLIER, DISTORTION_MULTIPLIER);
        current_node.next = numeric_limits<uint8_t>::max();
        current_node.token = DCT_EOB_TOKEN;
      }
    }
  }

  uint8_t token_context = ( frame_sb.context().above.initialized() ? frame_sb.context().above.get()->has_nonzero() : 0 )
    + ( frame_sb.context().left.initialized() ? frame_sb.context().left.get()->has_nonzero() : 0 );

  for ( size_t i = 0; i < LEVELS; i++ ) {
    TrellisNode & node = trellis.at( first_index ).at( i );
    node.rate += costs_.token_costs.at( frame_sb.type() )
                                   .at( coefficient_to_band.at( first_index ) )
                                   .at( token_context )
                                   .at( node.token );

    node.cost = rdcost( node.rate, node.distortion, RATE_MULTIPLIER,
                        DISTORTION_MULTIPLIER );
  }

  // walking the minium path through trellis
  uint32_t min_cost = numeric_limits<uint32_t>::max();
  size_t min_choice;
  for ( size_t i = 0; i < LEVELS; i++ ) {
    if ( trellis.at( first_index ).at( i ).cost < min_cost ) {
      min_cost = trellis.at( first_index ).at( i ).cost;
      min_choice = i;
    }
  }

  size_t i;
  for ( i = first_index; i < 16; i++ ) {
    if ( trellis.at( i ).at( min_choice ).token == DCT_EOB_TOKEN ) {
      break;
    }

    frame_sb.mutable_coefficients().at( zigzag.at( i ) ) = trellis.at( i ).at( min_choice ).coeff;
    min_choice = trellis.at( i ).at( min_choice ).next;
  }

  for ( ; i < 16; i++ ) {
    frame_sb.mutable_coefficients().at( zigzag.at( i ) ) = 0;
  }
}

uint32_t Encoder::rdcost( uint32_t rate, uint32_t distortion,
                          uint32_t rate_multiplier,
                          uint32_t distortion_multiplier )
{
  return ( ( 128 + rate * rate_multiplier ) / 256 )
         + distortion * distortion_multiplier;
}

template<class FrameType>
void Encoder::optimize_probability_tables( FrameType & frame, const TokenBranchCounts & token_branch_counts )
{
  for ( unsigned int i = 0; i < BLOCK_TYPES; i++ ) {
    for ( unsigned int j = 0; j < COEF_BANDS; j++ ) {
      for ( unsigned int k = 0; k < PREV_COEF_CONTEXTS; k++ ) {
        for ( unsigned int l = 0; l < ENTROPY_NODES; l++ ) {
          const unsigned int false_count = token_branch_counts.at( i ).at( j ).at( k ).at( l ).first;
          const unsigned int true_count = token_branch_counts.at( i ).at( j ).at( k ).at( l ).second;

          const unsigned int prob = Encoder::calc_prob( false_count, false_count + true_count );

          assert( prob <= 255 );

          if ( prob > 0 and prob != decoder_state_.probability_tables.coeff_probs.at( i ).at( j ).at( k ).at( l ) ) {
            frame.mutable_header().token_prob_update.at( i ).at( j ).at( k ).at( l ) = TokenProbUpdate( true, prob );
          }
        }
      }
    }
  }
}

template<class FrameHeaderType, class MacroblockType>
void Encoder::optimize_prob_skip( Frame<FrameHeaderType, MacroblockType> & frame )
{
  size_t no_skip_count = 0;
  size_t total_count = 0;

  frame.mutable_macroblocks().forall(
    [&] ( MacroblockType & frame_mb )
    {
      bool has_nonzero = frame_mb.has_nonzero();
      no_skip_count += has_nonzero;
      total_count++;
    }
  );

  frame.mutable_header().prob_skip_false.reset( Encoder::calc_prob( no_skip_count, total_count ) );
}

template<class FrameType>
void Encoder::apply_best_loopfilter_settings( const VP8Raster & original,
                                              VP8Raster & reconstructed,
                                              FrameType & frame )
{
  uint8_t best_lf_level = 0;
  double best_ssim = -1.0;

  uint8_t min_lf_level = 0;
  uint8_t max_lf_level = 63;

  if ( loop_filter_level_.initialized() ) {
    if ( loop_filter_level_.get() > 0 ) {
      min_lf_level = loop_filter_level_.get() - 1;
    }
    else {
      min_lf_level = 0;
    }

    max_lf_level = min( 63u, loop_filter_level_.get() + 1u );
  }

  for ( uint8_t lf_level = min_lf_level; lf_level <= max_lf_level; lf_level++ ) {
    temp_raster().copy_from( reconstructed );

    frame.mutable_header().loop_filter_level = lf_level;

    decoder_state_.filter_adjustments.reset( frame.header() );

    frame.loopfilter( decoder_state_.segmentation, decoder_state_.filter_adjustments, temp_raster() );

    /* XXX This is taking too much time and is very inefficient. */
    double ssim = temp_raster().quality( original );

    if ( ssim > best_ssim ) {
      best_ssim = ssim;
      best_lf_level = lf_level;
    }
    else {
      break;
    }
  }

  frame.mutable_header().loop_filter_level = best_lf_level;
  decoder_state_.filter_adjustments.reset( frame.header() );

  frame.loopfilter( decoder_state_.segmentation, decoder_state_.filter_adjustments, reconstructed );

  encode_stats_.ssim.reset( best_ssim );
}

template<class FrameType>
void Encoder::apply_cv_loopfilter(const VP8Raster & original,
                                  VP8Raster & reconstructed,
                                  FrameType & frame)
{
  /* skip loop filtering completely for now */
  frame.mutable_header().loop_filter_level = 0;
  decoder_state_.filter_adjustments.reset(frame.header());
  frame.loopfilter(decoder_state_.segmentation,
                   decoder_state_.filter_adjustments, reconstructed);

  /* for backward compatibility, compute SSIM */
  double ssim = reconstructed.quality(original);
  encode_stats_.ssim.reset(ssim);
}

template<class FrameType>
FrameType & Encoder::encode_with_quantizer_search( const VP8Raster & raster,
                                                   const double minimum_ssim )
{
  int y_ac_qi_min = 0;
  int y_ac_qi_max = 127;

  QuantIndices quant_indices;

  bool found = false;
  size_t best_y_ac_qi = 0;

  while ( y_ac_qi_min <= y_ac_qi_max ) {
    quant_indices.y_ac_qi = ( y_ac_qi_min + y_ac_qi_max ) / 2;

    pair<FrameType &, double> encoded_frame = encode_raster<FrameType>( raster, quant_indices, true );

    double current_ssim = encoded_frame.second;

    if ( current_ssim >= minimum_ssim || ( y_ac_qi_min == y_ac_qi_max && not found ) ) {
      // this is a potential answer, let's save it
      found = true;
      best_y_ac_qi = quant_indices.y_ac_qi;
    }

    if ( y_ac_qi_min == y_ac_qi_max ) {
      break;
    }

    if ( current_ssim < minimum_ssim ) {
      y_ac_qi_max = quant_indices.y_ac_qi - 1;
    }
    else {
      y_ac_qi_min = quant_indices.y_ac_qi + 1;
    }
  }

  quant_indices.y_ac_qi = best_y_ac_qi;
  return encode_raster<FrameType>( raster, quant_indices, false ).first;
}

vector<uint8_t> Encoder::encode_with_quantizer( const VP8Raster & raster, const uint8_t y_ac_qi )
{
  if ( width() != raster.display_width() or height() != raster.display_height() ) {
    throw runtime_error( "scaling is not supported" );
  }

  QuantIndices quant_indices;
  quant_indices.y_ac_qi = y_ac_qi;

  if ( not has_state_ ) {
    has_state_ = true;
    return write_frame( encode_raster<KeyFrame>( raster, quant_indices ).first );
  }
  else {
    return write_frame( encode_raster<InterFrame>( raster, quant_indices ).first );
  }
}

Segmentation Encoder::create_segmentation(VP8Raster & raster,
                                          const QuantIndices & quant_indices,
                                          const string & bbox_path)
{
  const unsigned width_in_mb = VP8Raster::macroblock_dimension(raster.display_width());
  const unsigned height_in_mb = VP8Raster::macroblock_dimension(raster.display_height());

  Segmentation s(width_in_mb, height_in_mb);
  s.absolute_segment_adjustments = false; /* relative adjustments */
  // s.segment_filter_adjustments = {0, 0, 0, 0};

  uint8_t bg_segid = num_segments, fg_segid = num_segments;

  if (bg_qi_ and fg_qi_) {
    s.segment_quantizer_adjustments = {127, 64, 32, 16};
    s.thresholds = {0, 0, 0, 0};

    for (unsigned i = 0; i < num_segments; i++) {
      if (*bg_qi_ == s.segment_quantizer_adjustments.at(i)) { bg_segid = i; }
      if (*fg_qi_ == s.segment_quantizer_adjustments.at(i)) { fg_segid = i; }
    }
  } else if (bg_th_ and fg_th_) {
    s.segment_quantizer_adjustments = {16, 16, 16, 16};
    s.thresholds = {1000, 500, 100, 0};

    for (unsigned i = 0; i < num_segments; i++) {
      if (*bg_th_ == s.thresholds.at(i)) { bg_segid = i; }
      if (*fg_th_ == s.thresholds.at(i)) { fg_segid = i; }
    }
  }

  if (bg_segid >= num_segments or fg_segid >= num_segments) {
    throw runtime_error("Invalid segment IDs for background or foreground");
  }

  /* correct segment_quantizer_adjustments if adjustments are relative */
  if (not s.absolute_segment_adjustments) {
    for (unsigned i = 0; i < num_segments; i++) {
      s.segment_quantizer_adjustments.at(i) -= quant_indices.y_ac_qi;
    }
  }

  vector<vector<bool>> foreground(width_in_mb, vector<bool>(height_in_mb));
  s.map.fill(bg_segid);

  if (not bbox_path.empty()) {
    const int mb_side = VP8Raster::macroblock_side;
    const int expand_mb = 1; /* expand margins of each bounding box by one macroblock */

    /* read bboxes from csv and set segmentation map */
    ifstream csv_stream(bbox_path);
    for (string line; getline(csv_stream, line);) {
      vector<string> items = split(line, ",");

      /* CSV format: class name, confidence,
       * left (inclusive), top (inclusive), right (exclusive), bottom (exclusive) */
      if (items.size() != 6) {
        throw runtime_error("invalid CSV file: " + bbox_path);
      }

      /* convert pixel to macroblock index (inclusive) and expand */
      unsigned c_min = max(0, stoi(items[2]) / mb_side - expand_mb);
      unsigned r_min = max(0, stoi(items[3]) / mb_side - expand_mb);
      unsigned c_max = min((int)width_in_mb - 1, (stoi(items[4]) - 1) / mb_side + expand_mb);
      unsigned r_max = min((int)height_in_mb - 1, (stoi(items[5]) - 1) / mb_side + expand_mb);

      for (unsigned c = c_min; c <= c_max; c++) {
        for (unsigned r = r_min; r <= r_max; r++) {
          s.map.at(c, r) = fg_segid;
          foreground[c][r] = true;
        }
      }
    }

    /* black out the background if black_bg_ is set */
    if (black_bg_) {
      for (unsigned c = 0; c < width_in_mb; c++) {
        for (unsigned r = 0; r < height_in_mb; r++) {
          if (not foreground[c][r]) {
            raster.macroblock(c, r).black_out();
          }
        }
      }
    }
  } else {
    throw runtime_error("bbox_path is empty");
  }

  return s;
}

vector<uint8_t> Encoder::encode_for_cv(VP8Raster & raster, const string & bbox_path)
{
  if (encode_quality_ != CV_QUALITY) {
    throw runtime_error("encode quality must be CV_QUALITY");
  }

  if (width() != raster.display_width() or height() != raster.display_height()) {
    throw runtime_error("scaling is not supported");
  }

  QuantIndices quant_indices;
  quant_indices.y_ac_qi = 64;
  Segmentation segmentation = create_segmentation(raster, quant_indices, bbox_path);

  if (not has_state_) {
    has_state_ = true;
    return write_frame(encode_raster<KeyFrame>(raster, quant_indices,
                                               false, false, move(segmentation)).first);
  } else {
    return write_frame(encode_raster<InterFrame>(raster, quant_indices,
                                                 false, false, move(segmentation)).first);
  }
}

vector<uint8_t> Encoder::encode_with_minimum_ssim( const VP8Raster & raster, const double minimum_ssim )
{
  if ( width() != raster.display_width() or height() != raster.display_height() ) {
    throw runtime_error( "scaling is not supported" );
  }

  if ( not has_state_ ) {
    has_state_ = true;
    return write_frame( encode_with_quantizer_search<KeyFrame>( raster, minimum_ssim ) );
  }
  else {
    return write_frame( encode_with_quantizer_search<InterFrame>( raster, minimum_ssim ) );
  }
}

vector<uint8_t> Encoder::encode_with_target_size( const VP8Raster & raster, const size_t target_size ) {
  if ( width() != raster.display_width() or height() != raster.display_height() ) {
    throw runtime_error( "scaling is not supported" );
  }

  int y_qi_min = 4;
  int y_qi_max = 127;

  if ( last_y_ac_qi_.initialized() ) {
    const int radius = 16;

    if ( last_y_ac_qi_.get() - radius >= y_qi_min ) {
      y_qi_min = last_y_ac_qi_.get() - radius;
    }

    y_qi_max = min( y_qi_max, last_y_ac_qi_.get() + radius );
  }

  uint8_t best_y_qi = numeric_limits<uint8_t>::max();

  size_t estimated_size = target_size;

  while ( y_qi_min <= y_qi_max ) {
    size_t y_qi = ( y_qi_min + y_qi_max ) / 2;
    estimated_size = estimate_frame_size( raster, y_qi );

    if ( estimated_size <= target_size or ( y_qi_min == y_qi_max and best_y_qi == numeric_limits<uint8_t>::max() ) ) {
      best_y_qi = y_qi;

      y_qi_max = y_qi - 1;
    }
    else if ( estimated_size > target_size ) {
      y_qi_min = y_qi + 1;
    }
  }

  return encode_with_quantizer( raster, best_y_qi );
}

ProbabilityArray<num_segments> Encoder::calc_segment_tree_probs(const SegmentationMap & seg_map) const
{
  assert(num_segments == 4);

  SafeArray<unsigned, num_segments> counts {0, 0, 0, 0};
  seg_map.forall(
    [&counts](const uint8_t segment_id) {
      counts.at(segment_id)++;
    });

  unsigned left_size = counts.at(0) + counts.at(1);
  unsigned right_size = counts.at(2) + counts.at(3);

  ProbabilityArray<num_segments> branch_probs;
  branch_probs.at(0) = calc_prob(left_size, left_size + right_size);
  branch_probs.at(1) = calc_prob(counts.at(0), left_size);
  branch_probs.at(2) = calc_prob(counts.at(2), right_size);

  return branch_probs;
}

template<class FrameHeaderType>
void Encoder::set_header_update_segmentation(FrameHeaderType & frame_header,
                                             const Optional<Segmentation> & segmentation)
{
  if (segmentation) {
    UpdateSegmentation update_seg;

    /* update segmentation map */
    update_seg.update_mb_segmentation_map = true;

    /* set segment feature data only on key frames */
    if (frame_header.key_frame()) {
      update_seg.segment_feature_data.initialize();
      SegmentFeatureData & seg_feature = *update_seg.segment_feature_data;

      seg_feature.segment_feature_mode = segmentation->absolute_segment_adjustments;
      for (unsigned i = 0; i < num_segments; i++) {
        seg_feature.quantizer_update.at(i) = Signed<7>(
          segmentation->segment_quantizer_adjustments.at(i));
        seg_feature.loop_filter_update.at(i) = Signed<6>(
          segmentation->segment_filter_adjustments.at(i));
      }
    }

    /* calculate and update segment probabilities based on frequency of segments */
    const auto & seg_tree_probs = calc_segment_tree_probs(segmentation->map);

    update_seg.mb_segmentation_map.initialize();
    auto & update_seg_probs = *update_seg.mb_segmentation_map;
    for (unsigned i = 0; i < update_seg_probs.size(); i++) {
      update_seg_probs.at(i) = Unsigned<8>(seg_tree_probs.at(i));
    }

    frame_header.update_segmentation = move(update_seg);
  } else {
    frame_header.update_segmentation.clear();
  }
}

template <class FrameHeaderType, class MacroblockHeaderType>
void Macroblock<FrameHeaderType, MacroblockHeaderType>::calculate_has_nonzero()
{
  has_nonzero_ = false;

  if ( Y2_.coded() ) {
    Y2_.calculate_has_nonzero();
    has_nonzero_ |= Y2_.has_nonzero();
  }

  Y_.forall( [&]( YBlock & block ) {
      block.calculate_has_nonzero();
      has_nonzero_ |= block.has_nonzero();
    } );

  /* parse U and V blocks */
  U_.forall( [&]( UVBlock & block ) {
      block.calculate_has_nonzero();
      has_nonzero_ |= block.has_nonzero();
    } );

  V_.forall( [&]( UVBlock & block ) {
      block.calculate_has_nonzero();
      has_nonzero_ |= block.has_nonzero();
    } );

  mb_skip_coeff_.reset( not has_nonzero_ );
}

template <class FrameHeaderType, class MacroblockHeaderType>
void Macroblock<FrameHeaderType, MacroblockHeaderType>::zero_out()
{
  if ( Y2_.coded() ) {
    Y2_.zero_out();
  }

  Y_.forall( [&]( YBlock & block ) {
      block.zero_out();
    } );

  /* parse U and V blocks */
  U_.forall( [&]( UVBlock & block ) {
      block.zero_out();
    } );

  V_.forall( [&]( UVBlock & block ) {
      block.zero_out();
    } );

  has_nonzero_ = false;
  mb_skip_coeff_.reset( true );
}


template void InterFrameMacroblock::zero_out();
template void KeyFrameMacroblock::zero_out();

Quantizers::Quantizers(const QuantIndices & quant_indices,
                       const Optional<Segmentation> & segmentation)
  : frame_quant_indices(quant_indices), frame_quantizer(quant_indices),
    segment_quant_indices(), segment_quantizers()
{
  if (segmentation) {
    segment_quant_indices.initialize();
    segment_quantizers.initialize();

    for (uint8_t i = 0; i < num_segments; i++) {
      auto & indices_i = segment_quant_indices->at(i);
      indices_i.y_ac_qi = segmentation->segment_quantizer_adjustments.at(i)
        + (segmentation->absolute_segment_adjustments
           ? static_cast<Unsigned<7>>(0)
           : quant_indices.y_ac_qi);

      segment_quantizers->at(i) = Quantizer(indices_i);
    }
  }
}

const Quantizer & Quantizers::get_quantizer(const uint8_t segment_id) const
{
  if (segment_id < num_segments and segment_quantizers) {
    return segment_quantizers->at(segment_id);
  }

  return frame_quantizer;
}

const QuantIndices & Quantizers::get_quant_indices(const uint8_t segment_id) const
{
  if (segment_id < num_segments and segment_quant_indices) {
    return segment_quant_indices->at(segment_id);
  }

  return frame_quant_indices;
}
