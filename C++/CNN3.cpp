#include <iostream>
#include <fstream>
#include <iomanip>
#include <math.h>
#include <vector>  
#include <random>
#include <thread>  
#include <boost/multi_array.hpp>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

using namespace std;
namespace p = boost::python;
namespace np = boost::python::numpy;

#include "Layers.cpp"

// precision for floating numbers written in files
constexpr int prec_write_file = 16; 

// define a standard normal distribution
std::random_device rd;  
std::mt19937 gen(rd()); 
std::normal_distribution<> dis(0.,1.);

// uniform dictribution
std::uniform_real_distribution<> uni(0.,1.);

struct double_and_int{
	double d = 0.;
	int i = 0;
};

class CNN3{

	private: 

		int num_CLs; // number of convolution layers
		int num_FCs; // number of FullCon layers
		int img_h_i; // image height
		int img_w_i; // image width
		int img_h; // image height before the fully connected layers
		int img_w; // image width before the fully connected layers
		int num_images; // number of images
		int num_labels; // number of different possible labels
		int n_channels; // number of channels
		
		vector<int> CL_size_filters; // Convolution layers: filter sizes
		vector<int> CL_num_filters; // Convolution layers: numbers of filters
		vector<int> MP_size; // Maxpool layers: pool sizes
		vector<int> FC_size; // FullCon layers: number of neurons
		
		// These four vectors will contain the layers
		vector<ConvLayer> CLs; // Convolution layers
		vector<ReLU> RLUs; // ReLU layers
		vector<MaxPool> MPs; // Maxpool layers
		vector<FullCon> FCs; // FullCon layers
		vector<SoftMax> SMs; // Softmax layers

	public: 

		CNN3(){
			np::initialize(); // required to create numpy arrays (otherwise leads to segmentation faults)
		}	

		CNN3(
			int img_w_, // image width 
			int img_h_, // image height
			int n_channels_, // number of channels
			p::list& CL_size_filters_, // list of filter sizes
			p::list& CL_num_filters_, // list of numbers of filters
			p::list& MP_size_, // list of pool sizes
			p::list& FC_size_, // list of FullCon sizes
			int num_labels_ // number of different possible labels
		) 
		{
			// initialization
			img_w_i = img_w_;
			img_h_i = img_h_;
			img_w = img_w_;
			img_h = img_h_;
			num_labels = num_labels_;
			n_channels = n_channels_;
			CL_size_filters = int_list_to_vector(CL_size_filters_);
			CL_num_filters = int_list_to_vector(CL_num_filters_);
			MP_size = int_list_to_vector(MP_size_);
			FC_size = int_list_to_vector(FC_size_);
			num_CLs = len(CL_size_filters_);
			num_FCs = len(FC_size_);
			
			num_images = n_channels; // tracks the number of images

			// build the layers
			for(int i=0; i<num_CLs; i++){
				CLs.push_back(ConvLayer(CL_size_filters[i], num_images, CL_num_filters[i], gen, dis));
				num_images = CL_num_filters[i];
				img_w = img_w + 1 - CL_size_filters[i];
				img_h = img_h + 1 - CL_size_filters[i];
				RLUs.push_back(ReLU());
				MPs.push_back(MaxPool(MP_size[i]));
				img_w = (int) img_w / MP_size[i];
				img_h = (int) img_h / MP_size[i];
			}
			long int n_inputs = num_images*img_h*img_w;
			for(int i=0; i<num_FCs; i++){
				int n_neurons = FC_size[i];
				FCs.push_back(FullCon(n_inputs, n_neurons, gen, dis));
				n_inputs = n_neurons;
			}
			SMs.push_back(SoftMax(n_inputs, num_labels, gen, dis));

			np::initialize(); // required to create numpy arrays (otherwise leads to segmentation faults)
		}

		// save the CNN parameters to a file
		void save(char* filename){
			ofstream file;
			file.open(filename);
			file << fixed << setprecision(prec_write_file);
			file << num_CLs << sep_val << num_FCs << sep_val << n_channels << sep_val << img_w_i << sep_val << img_h_i << sep_val << img_w << sep_val << img_h << sep_val << num_images << sep_line;
			save_vector(CL_size_filters, file);
			save_vector(CL_num_filters, file);
			save_vector(MP_size, file); 
			for(int i=0; i<num_CLs; i++){
				CLs[i].save(file);
				RLUs[i].save(file);
				MPs[i].save(file);
			}
			for(int i=0; i<num_FCs; i++){
				FCs[i].save(file);
			}
			SMs[0].save(file);
			file.close();
		}
		
		// load the CNN parameters from a file
		void load(char* filename){
			ifstream file;
			file.open(filename);
			char c;
			file >> num_CLs >> c >> num_FCs >> c >> n_channels >> c >> img_w_i >> c >> img_h_i >> c >> img_w >> c >> img_h >> c >> num_images >> c;
			CL_size_filters = load_vector<int>(file);
			CL_num_filters = load_vector<int>(file);
			MP_size = load_vector<int>(file); 
			CLs.clear();
			RLUs.clear();
			CLs.clear();
			SMs.clear();
			for(int i=0; i<num_CLs; i++){
				CLs.push_back(ConvLayer());
				CLs[i].load(file);
				RLUs.push_back(ReLU());
				RLUs[i].load(file);
				MPs.push_back(MaxPool());
				MPs[i].load(file);
			}
			for(int i=0; i<num_FCs; i++){
				FCs.push_back(FullCon());
				FCs[i].load(file);
			}
			SMs.push_back(SoftMax());
			SMs[0].load(file);
			file.close();
            num_labels = SMs[0].output.size();
		}

		// forward pass
		vector<double> forward(d3_array_type &input, double p_dropout = 0.){
			
			// number and dimensions of images
			int nim = input.shape()[0];
			int h = input.shape()[1];
			int w = input.shape()[2];
			if(h != img_h_i || w != img_w_i || nim != n_channels){
				cout << "\nInvalid input dimensions!\n" << endl;
			}
			for(int i=0; i<num_CLs; i++){
				d3_array_type output1 = CLs[i].forward(input);
				input.resize(boost::extents[output1.shape()[0]][output1.shape()[1]][output1.shape()[2]]);
				input = output1;

				d3_array_type output2 = MPs[i].forward(input);
				input.resize(boost::extents[output2.shape()[0]][output2.shape()[1]][output2.shape()[2]]);
				input = output2;

				d3_array_type output3 = RLUs[i].forward(input);
				input.resize(boost::extents[output3.shape()[0]][output3.shape()[1]][output3.shape()[2]]);
				input = output3;
			}
		
			vector<double> input_vec;
			auto input_shape = input.shape();
			for(int i=0; i<num_images; i++){
				for(int j=0; j<img_h; j++){
					for(int k=0; k<img_w; k++){
						input_vec.push_back(input[i][j][k]);
					}
				}
			}
			
			for(int i=0; i<num_FCs; i++){
				input_vec = FCs[i].forward(input_vec, gen, uni, p_dropout);
			}
			
			return SMs[0].forward(input_vec);
		}
		
		// backpropagation
		void backprop(vector<double> d_L_d_out_i, double learn_rate){
			
			d3_array_type d_L_d_in(boost::extents[num_images][img_h][img_w]); 
		
			vector<double> d_L_d_in_vec = SMs[0].backprop(d_L_d_out_i, learn_rate);
			
			for(int i=num_FCs-1; i>=0; i--){
				d_L_d_in_vec = FCs[i].backprop(d_L_d_in_vec, learn_rate);
			}
			
			for(int i=0; i<num_images; i++){
				for(int j=0; j<img_h; j++){
					for(int k=0; k<img_w; k++){
						d_L_d_in[i][j][k] = d_L_d_in_vec[i*img_h*img_w + j*img_w + k];
					}
				}
			}
			
			for(int i=num_CLs-1; i>=0; i--){
				d3_array_type d_L_d_in3 = RLUs[i].backprop(d_L_d_in);
				auto shape = d_L_d_in3.shape();
				d_L_d_in.resize(boost::extents[shape[0]][shape[1]][shape[2]]);
				d_L_d_in = d_L_d_in3;
			
				d3_array_type d_L_d_in2 = MPs[i].backprop(d_L_d_in);
				shape = d_L_d_in2.shape();
				d_L_d_in.resize(boost::extents[shape[0]][shape[1]][shape[2]]);
				d_L_d_in = d_L_d_in2;
				
				d3_array_type d_L_d_in1 = CLs[i].backprop(d_L_d_in, learn_rate);
				shape = d_L_d_in1.shape();
				d_L_d_in.resize(boost::extents[shape[0]][shape[1]][shape[2]]);
				d_L_d_in = d_L_d_in1;
			}
		}
		
		// loss function and accuracy (1 if correct answer, 0 otherwise)
		double_and_int loss_acc(vector<double> &output, int label){
			double_and_int results;
			results.d = -log(output[label]); 
			results.i = 1;
			for(int i=0; i<output.size(); i++){
				if(output[i] > output[label]){
					results.i = 0;
				}
			}
			return results;
		}

		// Completes a training step on the image 'image' with label 'label'.
		// Returns the corss-entropy and accuracy.
		double_and_int train(d3_array_type image, int label, double learn_rate = 0.005, double p_dropout = 0.) {
		
			// forward pass
			vector<double> output_forward = forward(image, p_dropout);
	
			// gradient of the loss function with respect to the output
			vector<double> d_L_d_out;
			for(int i=0; i<num_labels; i++){
				d_L_d_out.push_back(0.);
			}
			d_L_d_out[label] = -1./output_forward[label];

			// backpropagation
			backprop(d_L_d_out, learn_rate);

			// return loss and accuracy
			return loss_acc(output_forward, label);
		}

		// full forward propagation - Python wrapper
		// input: 2d numpy array
		np::ndarray forward_python(np::ndarray image){
			d3_array_type input = d3_numpy_to_multi_array(image);
			return vector_to_numpy(forward(input));
		}

		// forward - return loss and accuracy - Python wrapper
		p::list forward_la_python(np::ndarray image, int label){
			d3_array_type input = d3_numpy_to_multi_array(image);
			vector<double> output = forward(input);
			double_and_int results = loss_acc(output, label);
			p::list results_p;
			results_p.append(results.d);
			results_p.append(results.i);
			return results_p;
		}
		
		// full backpropagation - Python wrapper
		void backprop_python(np::ndarray d_L_d_out, double learn_rate){
			backprop(numpy_to_vector(d_L_d_out), learn_rate);
		}
		
		// train - Python wrapper
		p::list train_python(np::ndarray image, int label, double learn_rate, double p_dropout) {
			double_and_int results;
			results = train(d3_numpy_to_multi_array(image), label, learn_rate, p_dropout);
			p::list results_p;
			results_p.append(results.d);
			results_p.append(results.i);
			return results_p;
		}

};

BOOST_PYTHON_MODULE(CNN3)
{
    p::class_<CNN3>("CNN3", p::init<int, int, int, p::list&, p::list&, p::list&, p::list&, int>())
		.def(p::init<>())
		.def("forward", &CNN3::forward_python)
		.def("backprop", &CNN3::backprop_python)
		.def("train", &CNN3::train_python)
		.def("save", &CNN3::save)
		.def("load", &CNN3::load)
		.def("forward_la", &CNN3::forward_la_python)
	;
}
