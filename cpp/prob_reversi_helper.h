/*
    prob_reversi.pyのヘルパー関数.
    着手可能位置の計算や裏返る石の計算はコストがかかるので, C++側で実装する.
    大して規模の大きいプログラムではないので, 盤面のサイズや計算に必要なテーブルは全てグローバル変数で管理する.
*/

#pragma once

#include <cstdint>

class __Helper
{
    public:
        __Helper(std::int32_t board_size);

        /**
         * @fn
         * @brief 
         *      着手可能位置を求める.
         * @param p 
         *      手番の石の配置.
         * @param o 
         *      相手の石の配置.
         * @return
         *      着手可能位置が1となっているビットボード.
         */
        std::uint64_t calc_mobility(std::uint64_t p, std::uint64_t o);

        /**
         * @fn
         * @brief 
         *      裏返る石を求める.
         * @param p 
         *      手番の石の配置.
         * @param o 
         *      相手の石の配置.
         * @param coord
         *      着手する場所.
         * @return
         *      裏返る石の位置が1となっているビットボード.
         */
        std::uint64_t calc_flip_discs(std::uint64_t p, std::uint64_t o, std::int32_t coord);

    private:
        std::int32_t board_size;    // 盤面のサイズ.
        std::int32_t square_num;   // マス目の数.
        std::uint64_t valid_bits_mask;   // 使用するビットだけ1が立っているマスク. 例えば, 4x4盤面であれば, 下位16bitのみ用いるので, 0x000000000000ffffULL となる.
        std::int32_t shift_table[4];    // 盤面の各方向に対するシフト数.
        std::uint64_t masks[4];    // 盤面の各方向用のビットマスク. 計算時に盤面の上下左右が繋がることを防ぐ.
};
