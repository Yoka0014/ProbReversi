/*
    prob_reversi.pyのヘルパー関数.
    着手可能位置の計算や裏返る石の計算はコストがかかるので, C++側で実装する.
    大して規模の大きいプログラムではないので, 盤面のサイズや計算に必要なテーブルは全てグローバル変数で管理する.
*/

#pragma once

#include <cstdint>

extern std::int32_t g_board_size;    // 盤面のサイズ.
extern std::int32_t g_square_num;   // マス目の数.
extern std::uint64_t g_valid_bits_mask;   // 使用するビットだけ1が立っているマスク. 例えば, 4x4盤面であれば, 下位16bitのみ用いるので, 0x000000000000ffffULL となる.
extern std::int32_t g_shift_table[4];    // 盤面の各方向に対するシフト数.
extern std::uint64_t g_masks[4];    // 盤面の各方向用のビットマスク. 計算時に盤面の上下左右が繋がることを防ぐ.

bool __set_board_size(std::int32_t size);
std::uint64_t __calc_mobility(std::uint64_t p, std::uint64_t o);
std::uint64_t __calc_flip_discs(std::uint64_t p, std::uint64_t o, std::int32_t coord);
 