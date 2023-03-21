#include "prob_reversi_helper.h"

#include <cstring>

using namespace std;

__Helper::__Helper(int32_t size) : board_size(size), square_num(size * size), shift_table(), masks()
{
    this->valid_bits_mask = (this->square_num < 64) ? (1ULL << this->square_num) - 1 : 0xffffffffffffffffULL;
    this->shift_table[0] = 1;
    this->shift_table[1] = size;
    this->shift_table[2] = size + 1;
    this->shift_table[3] = size - 1;

    auto vm = ~((1ULL << (size - 1)) + 1ULL) & ((1ULL << size) - 1);
    auto v_mask = 0ULL;
    for (auto i = 0; i < size; i++)
    {
        v_mask |= vm;
        vm <<= size;
    }

    this->masks[0] = this->masks[2] = this->masks[3] = v_mask;
    this->masks[1] = this->valid_bits_mask;
}

uint64_t __Helper::calc_mobility(uint64_t p, uint64_t o)
{
    int32_t size = this->board_size;
    int32_t* shift_table = this->shift_table;
    uint64_t* masks = this->masks;

    uint64_t mobility[4]{}; // ループ前後の依存関係を無くすために, mobilityは方向ごとに配列に格納.
    for (auto dir = 0; dir < 4; dir++)  // 左方向にチェック.
    {
        int32_t shift = shift_table[dir];
        uint64_t masked_o = o & masks[dir];
        uint64_t flip = (p << shift) & masked_o;
        for(auto i = 0; i < size - 2; i++)
            flip |= (flip << shift) & masked_o;
        mobility[dir] |= flip << shift;
    }

    for (auto dir = 0; dir < 4; dir++)  // 右方向にチェック.
    {
        int32_t shift = shift_table[dir];
        uint64_t masked_o = o & masks[dir];
        uint64_t flip = (p >> shift) & masked_o;
        for(auto i = 0; i < size - 2; i++)
            flip |= (flip >> shift) & masked_o;
        mobility[dir] |= flip >> shift;
    }

    return (mobility[0] | mobility[1] | mobility[2] | mobility[3]) & ~(p | o) & this->valid_bits_mask;
}

uint64_t __Helper::calc_flip_discs(uint64_t p, uint64_t o, int32_t coord)
{
    int32_t size = this->board_size;
    int32_t* shift_table = this->shift_table;
    uint64_t* masks = this->masks;
    auto x = 1ULL << coord;

    uint64_t flip[4]{}; // ループ前後の依存関係を無くすために, flipは方向ごとに配列に格納.
    for (auto dir = 0; dir < 4; dir++)  // 左方向にチェック.
    {
        int32_t shift = shift_table[dir];
        uint64_t masked_o = o & masks[dir];
        uint64_t flip_l = (x << shift) & masked_o;
        for(auto i = 0; i < size - 2; i++)
            flip_l |= (flip_l << shift) & masked_o;
        uint64_t outflank = p & (flip_l << shift);  // 連続した相手石の隣にある自石.
        flip[dir] |= (flip_l & -static_cast<int32_t>(outflank != 0ULL));    // 連続した相手石の隣にある自石があれば, 裏返る石としてflip[dir]に追加. 
    }

    for (auto dir = 0; dir < 4; dir++)  // 右方向にチェック.
    {
        int32_t shift = shift_table[dir];
        uint64_t masked_o = o & masks[dir];
        uint64_t flip_r = (x >> shift) & masked_o;
        for(auto i = 0; i < size - 2; i++)
            flip_r |= (flip_r >> shift) & masked_o;
        uint64_t outflank = p & (flip_r >> shift);  
        flip[dir] |= (flip_r & -static_cast<int32_t>(outflank != 0ULL));    
    }

    return flip[0] | flip[1] | flip[2] | flip[3];
}