//blob的基本使用

#include <iostream>
#include "caffe/blob.hpp"


int main111(int argc, char* argv[])
{
	//构造一个blob
	caffe::Blob<float> b;
	std::cout << "Size:" << b.shape_string() << std::endl;
	b.Reshape(1, 2, 3, 4);
	std::cout << "Size:" << b.shape_string() << std::endl;

	//使用mutable_cpu_data函数修改Blob内部数值
	float* p = b.mutable_cpu_data();
	float* q = b.mutable_cpu_diff();
	for (int i = 0; i < b.count();i++)
	{
		p[i] = i;//data初始化为0,1,2,
		q[i] = b.count() - 1 - i;//将diff初始化为23,22，21......
	}

	//打印指定位置的每一个数值
	std::cout << "data和diff融合之前" << std::endl;
	for (int u = 0; u < b.num();u++)
	{
		for (int v = 0; v < b.channels();v++)
		{
			for (int w = 0; w < b.height();w++)
			{
				for (int x = 0; x < b.width();x++)
				{
					std::cout << "b[" << u << "][" << v << "][" << w << "][" << x << "]=";
				}
			}
		}
	}
	//求L1和L2范式及其输出结果
	std::cout << "ASUM:" << b.asum_data() << std::endl;
	std::cout << "SUMSQ:" << b.sumsq_data() << std::endl;
	
	
	b.Update();
	std::cout << "data和diff融合之后" << std::endl;
	for (int u = 0; u < b.num(); u++)
	{
		for (int v = 0; v < b.channels(); v++)
		{
			for (int w = 0; w < b.height(); w++)
			{
				for (int x = 0; x < b.width(); x++)
				{
					std::cout << "b[" << u << "][" << v << "][" << w << "][" << x << "]=";
				}
			}
		}
	}
	char c = getchar();
	return 0;

}