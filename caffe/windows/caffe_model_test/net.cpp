//net的基本用法
#include <iostream>
#include <caffe/net.hpp>
#include <vector>
#include <caffe/layers/input_layer.hpp>

using namespace caffe;
using namespace std;

int main123()
{
	string proto("D://WorkSpace//caffe//caffe//models//bvlc_alexnet//deploy.prototxt");
	Net<float> n(proto, TEST);
	vector<string> blob_name = n.blob_names();
	vector<string> layer_name = n.layer_names();
	cout << "layer names:" << endl;

	for (int i = 0; i < layer_name.size(); i++)
	{
		cout << "layer #" << i << ":" << layer_name[i] << endl;
	}

	
	cout << "blob names:" << endl;

	for (int i = 0; i < blob_name.size(); i++)
	{
		cout << "Blob #" << i << ":" << blob_name[i] << endl;
	}
	cout << "是否有conv6这个层？" << endl;
	cout << n.has_layer("conv6") << endl;
	return 0;

}