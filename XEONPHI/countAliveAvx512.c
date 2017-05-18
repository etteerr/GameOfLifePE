size_t countAliveIntAVX512(int *map, struct vector2u size) {
    /*
     * Important:
     *  We are addind into 4*16 accumulators of size int.
     *  Dont add them together as int max may not be sufficent!
     *  Accumulate them to size_t!
     */
    int accs[4*16];
    size_t i,elems = size.x * size.y, count = 0;
    register __m512i acc1, acc2, acc3, acc4; //D: 4x 512bits D:
    acc1 = acc2 = acc3 = acc4 = _mm512_setzero_epi32();
    
    for(i=0; i<elems; i+=64) {
        acc1 = _mm512_add_epi32(acc1,_mm512_load_epi32(map+i+0));
        acc2 = _mm512_add_epi32(acc2,_mm512_load_epi32(map+i+16));
        acc3 = _mm512_add_epi32(acc3,_mm512_load_epi32(map+i+32));
        acc4 = _mm512_add_epi32(acc4,_mm512_load_epi32(map+i+48));
    }
    
    for(i-=63; i<elems; i++)
        count += map[i];
    
    _mm512_store_epi32(&accs[0], acc1);
    _mm512_store_epi32(&accs[16], acc2);
    _mm512_store_epi32(&accs[32], acc3);
    _mm512_store_epi32(&accs[48], acc4);
    
    for(i=0; i<64; i+=4){
        count += accs[i+0];
        count += accs[i+1];
        count += accs[i+2];
        count += accs[i+3];
    }
} 
