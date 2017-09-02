//blob�Ļ���ʹ��

#include <iostream>
#include "caffe/blob.hpp"


int main111(int argc, char* argv[])
{
	//����һ��blob
	caffe::Blob<float> b;
	std::cout << "Size:" << b.shape_string() << std::endl;
	b.Reshape(1, 2, 3, 4);
	std::cout << "Size:" << b.shape_string() << std::endl;

	//ʹ��mutable_cpu_data�����޸�Blob�ڲ���ֵ
	float* p = b.mutable_cpu_data();
	float* q = b.mutable_cpu_diff();
	for (int i = 0; i < b.count();i++)
	{
		p[i] = i;//data��ʼ��Ϊ0,1,2,
		q[i] = b.count() - 1 - i;//��diff��ʼ��Ϊ23,22��21......
	}

	//��ӡָ��λ�õ�ÿһ����ֵ
	std::cout << "data��diff�ں�֮ǰ" << std::endl;
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
	//��L1��L2��ʽ����������
	std::cout << "ASUM:" << b.asum_data() << std::endl;
	std::cout << "SUMSQ:" << b.sumsq_data() << std::endl;
	
	
	b.Update();
	std::cout << "data��diff�ں�֮��" << std::endl;
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