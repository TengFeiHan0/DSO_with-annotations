#include "frontend/PixelSelector2.h"
#include "internal/FrameHessian.h"
#include "internal/GlobalCalib.h"

using namespace ldso::internal;

namespace ldso {

    PixelSelector::PixelSelector(int w, int h) {
        randomPattern = new unsigned char[w * h];
        std::srand(3141592);    // want to be deterministic.
        for (int i = 0; i < w * h; i++)
            randomPattern[i] = rand() & 0xFF;
        currentPotential = 3;
        gradHist = new int[100 * (1 + w / 32) * (1 + h / 32)];
        ths = new float[(w / 32) * (h / 32) + 100];
        thsSmoothed = new float[(w / 32) * (h / 32) + 100];
    }

    PixelSelector::~PixelSelector() {
        delete[] randomPattern;
        delete[] gradHist;
        delete[] ths;
        delete[] thsSmoothed;
    }

    int computeHistQuantil(int *hist, float below) {
        int th = hist[0] * below + 0.5f;
        for (int i = 0; i < 90; i++) {
            th -= hist[i + 1];
            if (th < 0) return i;
        }
        return 90;
    }
/*
    *将图像划分为固定的 block, 每个 block 大小固定为 32*32
    * 在每个 block 内创建直方图 hist0
    * 统计直方图 hist0 中像素数目占百分之 50 对应的梯度作为阈值 ths
    * 对阈值进行 3*3 的均值滤波，得到 thsSmoothed
    * 
    *
*/
    void PixelSelector::makeHists(shared_ptr<FrameHessian> fh) {
        gradHistFrame = fh;
        float *mapmax0 = fh->absSquaredGrad[0];

        int w = wG[0];
        int h = hG[0];

        int w32 = w / 32;
        int h32 = h / 32;
        thsStep = w32;

        for (int y = 0; y < h32; y++)
            for (int x = 0; x < w32; x++) {
                float *map0 = mapmax0 + 32 * x + 32 * y * w;
                int *hist0 = gradHist;// + 50*(x+y*w32);
                memset(hist0, 0, sizeof(int) * 50);

                for (int j = 0; j < 32; j++)
                    for (int i = 0; i < 32; i++) {
                        int it = i + 32 * x;
                        int jt = j + 32 * y;
                        if (it > w - 2 || jt > h - 2 || it < 1 || jt < 1) continue;
                        int g = sqrtf(map0[i + j * w]);
                        if (g > 48) g = 48;
                        hist0[g + 1]++;
                        hist0[0]++;
                    }
                // 得到每一block的阈值
                ths[x + y * w32] = computeHistQuantil(hist0, setting_minGradHistCut) + setting_minGradHistAdd;
            }
        // 使用3*3的窗口求平均值来平滑
        for (int y = 0; y < h32; y++)
            for (int x = 0; x < w32; x++) {
                float sum = 0, num = 0;
                if (x > 0) {
                    if (y > 0) {
                        num++;
                        sum += ths[x - 1 + (y - 1) * w32];
                    }
                    if (y < h32 - 1) {
                        num++;
                        sum += ths[x - 1 + (y + 1) * w32];
                    }
                    num++;
                    sum += ths[x - 1 + (y) * w32];
                }

                if (x < w32 - 1) {
                    if (y > 0) {
                        num++;
                        sum += ths[x + 1 + (y - 1) * w32];
                    }
                    if (y < h32 - 1) {
                        num++;
                        sum += ths[x + 1 + (y + 1) * w32];
                    }
                    num++;
                    sum += ths[x + 1 + (y) * w32];
                }

                if (y > 0) {
                    num++;
                    sum += ths[x + (y - 1) * w32];
                }
                if (y < h32 - 1) {
                    num++;
                    sum += ths[x + (y + 1) * w32];
                }
                num++;
                sum += ths[x + y * w32];

                thsSmoothed[x + y * w32] = (sum / num) * (sum / num);
            }
    }
/*
 * @ function:
 * 
 * @ param: 	fh				帧Hessian数据结构
 * @			map_out			选出的地图点
 * @			density		 	每一金字塔层要的点数(密度)
 * @			recursionsLeft	最大递归次数
 * @			plot			画图
 * @			thFactor		阈值因子
 * @
 * @ note:		使用递归
 */
    int PixelSelector::makeMaps(const shared_ptr<FrameHessian> fh, float *map_out, float density,
                                int recursionsLeft, bool plot, float thFactor) {

        float numHave = 0;
        float numWant = density;
        float quotia;
        int idealPotential = currentPotential;

        if (fh != gradHistFrame) makeHists(fh);//!步骤1，生成梯度直方图

        //!步骤2， select!选择合适的像素点
        Eigen::Vector3i n = this->select(fh, map_out, currentPotential, thFactor);

        // sub-select!
        numHave = n[0] + n[1] + n[2];
        quotia = numWant / numHave;//!想要的  得到的  比例
        
        //! 步骤3，对阈值进行 3*3 的均值滤波，得到 thsSmoothed
        // by default we want to over-sample by 40% just to be sure.
        // 相当于覆盖的面积, 每一个像素对应一个pot*pot
        float K = numHave * (currentPotential + 1) * (currentPotential + 1);
        idealPotential = sqrtf(K / numWant) - 1;    // round down.
        if (idealPotential < 1) idealPotential = 1;
        //!步骤4，想要的数目和已经得到的数目, 大于或小于0.25都会重新采样一次
        if (recursionsLeft > 0 && quotia > 1.25 && currentPotential > 1) {
            // re-sample to get more points!
            // potential needs to be smaller
            if (idealPotential >= currentPotential)
                idealPotential = currentPotential - 1;

            currentPotential = idealPotential;
            return makeMaps(fh, map_out, density, recursionsLeft - 1, plot, thFactor);
        } else if (recursionsLeft > 0 && quotia < 0.25) {
            // re-sample to get less points!
            if (idealPotential <= currentPotential)
                idealPotential = currentPotential + 1;
            currentPotential = idealPotential;
            return makeMaps(fh, map_out, density, recursionsLeft - 1, plot, thFactor);
        }
        //!步骤5，点太多，随机剔除
        int numHaveSub = numHave;
        if (quotia < 0.95) {
            int wh = wG[0] * hG[0];
            int rn = 0;
            unsigned char charTH = 255 * quotia;
            for (int i = 0; i < wh; i++) {
                if (map_out[i] != 0) {
                    if (randomPattern[rn] > charTH) {
                        map_out[i] = 0;
                        numHaveSub--;
                    }
                    rn++;
                }
            }
        }

        currentPotential = idealPotential;

        return numHaveSub;
    }
/*
 * @ function:		根据阈值选择不同层上符合要求的像素
 * 
 * @ param: 		fh						帧的一些信息
 * @				map_out					选中的像素点及所在层
 * @				pot(currentPotential)	选点的范围大小, 一个pot内选一个
 * @				thFactor				阈值因子(乘数)
 * 
 * @ note:			返回的是每一层选择的点的个数
*/
    Eigen::Vector3i PixelSelector::select(const shared_ptr<FrameHessian> fh, float *map_out, int pot,
                                          float thFactor) {
        Eigen::Vector3f const *const map0 = fh->dI;

        float *mapmax0 = fh->absSquaredGrad[0];//第0层的梯度平方和
        float *mapmax1 = fh->absSquaredGrad[1];
        float *mapmax2 = fh->absSquaredGrad[2];


        int w = wG[0];
        int w1 = wG[1];
        int w2 = wG[2];
        int h = hG[0];

        //? 这个是为了什么呢, 
	    //! 随机选这16个对应方向上的梯度和阈值比较
	    //! 每个pot里面的方向随机选取的, 防止特征相同, 重复
	    // 模都是1
        const Vec2f directions[16] = {
                Vec2f(0, 1.0000),
                Vec2f(0.3827, 0.9239),
                Vec2f(0.1951, 0.9808),
                Vec2f(0.9239, 0.3827),
                Vec2f(0.7071, 0.7071),
                Vec2f(0.3827, -0.9239),
                Vec2f(0.8315, 0.5556),
                Vec2f(0.8315, -0.5556),
                Vec2f(0.5556, -0.8315),
                Vec2f(0.9808, 0.1951),
                Vec2f(0.9239, -0.3827),
                Vec2f(0.7071, -0.7071),
                Vec2f(0.5556, 0.8315),
                Vec2f(0.9808, -0.1951),
                Vec2f(1.0000, 0.0000),
                Vec2f(0.1951, -0.9808)};

        memset(map_out, 0, w * h * sizeof(PixelSelectorStatus));
        // 金字塔层阈值的减小倍数
        float dw1 = setting_gradDownweightPerLevel;
        float dw2 = dw1 * dw1;
        // 第2层1个pot对应第1层4个pot, 第1层1个pot对应第0层的4个pot,
	    // 第0层的4个pot里面只要选一个像素, 就不在对应高层的pot里面选了,
	    // 但是还会在第0层的每个pot里面选大于阈值的像素
	    // 阈值随着层数增加而下降
	    // 从顶层向下层遍历, 写的挺有意思!
        int n3 = 0, n2 = 0, n4 = 0;
        //* 第2层中, 每隔pot选一个点遍历
        for (int y4 = 0; y4 < h; y4 += (4 * pot))
            for (int x4 = 0; x4 < w; x4 += (4 * pot)) {
                int my3 = std::min((4 * pot), h - y4);
                int mx3 = std::min((4 * pot), w - x4);
                int bestIdx4 = -1;
                float bestVal4 = 0;
                //随机系数
                Vec2f dir4 = directions[randomPattern[n2] & 0xF];
                //* 上面的领域范围内, 在第1层进行遍历, 每隔pot一个点
                for (int y3 = 0; y3 < my3; y3 += (2 * pot))
                    for (int x3 = 0; x3 < mx3; x3 += (2 * pot)) {
                        int x34 = x3 + x4;
                        int y34 = y3 + y4;
                        int my2 = std::min((2 * pot), h - y34);
                        int mx2 = std::min((2 * pot), w - x34);
                        int bestIdx3 = -1;
                        float bestVal3 = 0;
                        Vec2f dir3 = directions[randomPattern[n2] & 0xF];
                        //* 上面的邻域范围内, 变换到第0层, 每隔pot遍历
			            //! 每个pot大小格里面一个大于阈值的最大的像素
                        for (int y2 = 0; y2 < my2; y2 += pot)
                            for (int x2 = 0; x2 < mx2; x2 += pot) {
                                int x234 = x2 + x34;
                                int y234 = y2 + y34;
                                int my1 = std::min(pot, h - y234);
                                int mx1 = std::min(pot, w - x234);
                                int bestIdx2 = -1;
                                float bestVal2 = 0;
                                Vec2f dir2 = directions[randomPattern[n2] & 0xF];
                                //* 第0层中的,pot大小邻域内遍历
                                for (int y1 = 0; y1 < my1; y1 += 1)
                                    for (int x1 = 0; x1 < mx1; x1 += 1) {
                                        assert(x1 + x234 < w);
                                        assert(y1 + y234 < h);
                                        int idx = x1 + x234 + w * (y1 + y234);
                                        int xf = x1 + x234;
                                        int yf = y1 + y234;

                                        if (xf < 4 || xf >= w - 5 || yf < 4 || yf > h - 4) continue;

                                        // 直方图求得阈值, 除以32确定在哪个阈值范围, 
					                    //! 可以确定是每个grid, 32格大小
                                        float pixelTH0 = thsSmoothed[(xf >> 5) + (yf >> 5) * thsStep];
                                        float pixelTH1 = pixelTH0 * dw1;
                                        float pixelTH2 = pixelTH1 * dw2;


                                        float ag0 = mapmax0[idx];
                                        if (ag0 > pixelTH0 * thFactor) {
                                            Vec2f ag0d = map0[idx].tail<2>();
                                            float dirNorm = fabsf((float) (ag0d.dot(dir2)));
                                            if (!setting_selectDirectionDistribution) dirNorm = ag0;

                                            if (dirNorm > bestVal2) {
                                                bestVal2 = dirNorm;
                                                bestIdx2 = idx;
                                                bestIdx3 = -2;
                                                bestIdx4 = -2;
                                            }
                                        }
                                        if (bestIdx3 == -2) continue;

                                        float ag1 = mapmax1[(int) (xf * 0.5f + 0.25f) + (int) (yf * 0.5f + 0.25f) * w1];
                                        if (ag1 > pixelTH1 * thFactor) {
                                            Vec2f ag0d = map0[idx].tail<2>();
                                            float dirNorm = fabsf((float) (ag0d.dot(dir3)));
                                            if (!setting_selectDirectionDistribution) dirNorm = ag1;

                                            if (dirNorm > bestVal3) {
                                                bestVal3 = dirNorm;
                                                bestIdx3 = idx;
                                                bestIdx4 = -2;
                                            }
                                        }
                                        if (bestIdx4 == -2) continue;

                                        float ag2 = mapmax2[(int) (xf * 0.25f + 0.125) +
                                                            (int) (yf * 0.25f + 0.125) * w2];
                                        if (ag2 > pixelTH2 * thFactor) {
                                            Vec2f ag0d = map0[idx].tail<2>();
                                            float dirNorm = fabsf((float) (ag0d.dot(dir4)));
                                            if (!setting_selectDirectionDistribution) dirNorm = ag2;

                                            if (dirNorm > bestVal4) {
                                                bestVal4 = dirNorm;
                                                bestIdx4 = idx;
                                            }
                                        }
                                    }

                                if (bestIdx2 > 0) {
                                    map_out[bestIdx2] = 1;
                                    bestVal3 = 1e10;
                                    n2++;
                                }
                            }

                        if (bestIdx3 > 0) {
                            map_out[bestIdx3] = 2;
                            bestVal4 = 1e10;
                            n3++;
                        }
                    }

                if (bestIdx4 > 0) {
                    map_out[bestIdx4] = 4;
                    n4++;
                }
            }


        return Eigen::Vector3i(n2, n3, n4);
    }

}