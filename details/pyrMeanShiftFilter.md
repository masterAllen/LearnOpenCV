
网上的文章真是离谱，我感觉作者也没懂就在那瞎写，白白浪费时间。后来看到[新浪博客 2012 年的一篇文章](https://blog.sina.com.cn/s/blog_63913ba601013336.html)，反而还可以，人家毕竟务实，直接看源代码，比那些不懂装懂的人好的太多。

还是先务虚，先用文字表达其作用：
> 以输入图像上src上任一点P0为圆心，建立物理空间上半径为sp，色彩空间上半径为sr的球形空间，物理空间上坐标2个—x、y，色彩空间上坐标3个—R、G、B（或HSV），构成一个5维的空间球体。构建的球形空间中，求得所有点相对于中心点的色彩向量之和后，移动迭代空间的中心点到该向量的终点，并再次计算该球形空间中所有点的向量之和，如此迭代，直到在最后一个空间球体中所求得的向量和的终点就是该空间球体的中心点Pn，迭代结束。

最好要知道 MeanShift 大致是什么意思。下面是源码，只把关键的部分弄出来，其他部分都忽略，比如表示金字塔层数的 max_level 就把他当成 0 来看，这样避免太多细节的干扰。源代码在 imgproc 中的 segmentation.cpp 里：
```cpp
// sp0: 空间半径、 sr：颜色半径
void cv::pyrMeanShiftFiltering( InputArray _src, OutputArray _dst, double sp0, double sr, int max_level, TermCriteria termcrit )
{
    const int cn = 3;
    const int MAX_LEVELS = 8;

    double sr2 = sr * sr;
    int isr2 = cvRound(sr2), isr22 = MAX(isr2,16);

    // 2. apply meanshift, starting from the pyramid top (i.e. the smallest layer)
    // 忽略这外层循环，看里面核心部分
    for( level = max_level; level >= 0; level-- )
    {
        // 这两个 for 循环就是遍历像素
        for( i = 0; i < size.height; i++, sptr += sstep - size.width*3, dptr += dstep - size.width*3)
        {
            for( j = 0; j < size.width; j++, sptr += 3, dptr += 3 )
            {
                // 现在开始，正式对一个像素处理。显然，c0-c2 就是 RGB 值呗
                c0 = sptr[0], c1 = sptr[1], c2 = sptr[2];

                // iterate meanshift procedure
                for( iter = 0; iter < termcrit.maxCount; iter++ )
                {
                    const uchar* ptr;
                    int minx, miny, maxx, maxy;

                    // mean shift: process pixels in window (p-sigmaSp)x(p+sigmaSp)
                    // x0, y0 就是当前中心点的位置，这部分就是计算出物理空间，考虑边界
                    minx = cvRound(x0 - sp); minx = MAX(minx, 0);
                    miny = cvRound(y0 - sp); miny = MAX(miny, 0);
                    maxx = cvRound(x0 + sp); maxx = MIN(maxx, size.width-1);
                    maxy = cvRound(y0 + sp); maxy = MIN(maxy, size.height-1);
                    ptr = sptr + (miny - i)*sstep + (minx - j)*3;

                    // 然后就在这个物理空间内计算
                    for( y = miny; y <= maxy; y++, ptr += sstep - (maxx-minx+1)*3 )
                    {
                        int row_count = 0;
                        x = minx;
                        // 删了一些东西，意会即可，相当于就是把物理空间内、又在颜色空间内的那些像素加起来
                        for( ; x <= maxx; x++, ptr += 3 )
                        {
                            int t0 = ptr[0], t1 = ptr[1], t2 = ptr[2];
                            // is2 就是最上面根据颜色半径算出的值，其实就是 sr*sr
                            if( tab[t0-c0+255] + tab[t1-c1+255] + tab[t2-c2+255] <= isr2 )
                            {
                                s0 += t0; s1 += t1; s2 += t2;
                                sx += x; row_count++;
                            }
                        }
                        count += row_count;
                        sy += y*row_count;
                    }

                    if( count == 0 )
                        break;

                    // 之前把像素都加起来，这里就是求均值呗
                    icount = 1./count;
                    x1 = cvRound(sx*icount); y1 = cvRound(sy*icount);
                    s0 = cvRound(s0*icount); s1 = cvRound(s1*icount); s2 = cvRound(s2*icount);

                    // 停止条件
                    stop_flag = (x0 == x1 && y0 == y1) || std::abs(x1-x0) + std::abs(y1-y0) +
                        tab[s0 - c0 + 255] + tab[s1 - c1 + 255] +
                        tab[s2 - c2 + 255] <= termcrit.epsilon;

                    x0 = x1; y0 = y1;
                    c0 = s0; c1 = s1; c2 = s2;

                    if( stop_flag )
                        break;
                }

                // 到这里就相当于跳出循环，可以赋值给目标指针了
                dptr[0] = (uchar)c0;
                dptr[1] = (uchar)c1;
                dptr[2] = (uchar)c2;
            }
        }
    }
}
```
