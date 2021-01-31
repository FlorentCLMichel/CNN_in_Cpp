FullCon::FullCon(int input_len, int output_len, 
	    	std::mt19937 &gen, std::normal_distribution<> &dis) : 
		weights(boost::extents[output_len][input_len])
{
	// initialize the biases to 0., the weights to random values, and keep to 1
	for(int i=0; i<output_len; i++){
		biases.push_back(0.);
		keep.push_back(0);
		for(int j=0; j<input_len; j++){
			weights[i][j] = dis(gen) / input_len;
		}
	}
}
		
void FullCon::save(ofstream& file){
	save_2d_array(weights, file);
	save_vector(biases, file);
	save_vector(totals, file);
	save_vector(last_input, file);
}

void FullCon::load(ifstream& file){
	d2_array_type weights_ = load_2d_array(file);
	auto shape_weights = weights_.shape();
	weights.resize(boost::extents[shape_weights[0]][shape_weights[1]]);
	weights = weights_;
	
	biases = load_vector<double>(file);
	
	totals = load_vector<double>(file);

	vector<double> last_input = load_vector<double>(file);

	keep = vector<int>(shape_weights[0], 1);
}

//// sigmoid function
//double sigmoid(double x){
//	return (1./(1.+exp(-x)));
//}
//
//// derivative of the sigmoid function
//double sigmoid_p(double x){
//	double s = sigmoid(x);
//	return s*(1.-s);
//}

// activation function 
double activation(double x){
	if (x >= 0.) {
		return x;
	} else {
		return 0.;
	}
}

// derivative of the activation function 
double activation_p(double x){
	if (x >= 0.) {
		return 1.;
	} else {
		return 0.;
	}
}

vector<double> FullCon::forward(vector<double> &input, std::mt19937 &gen, std::uniform_real_distribution<> &uni, double p_dropout = 0.){

	// clear the vector of totals
	totals.clear();

	vector<double> output;
	int n_nodes = weights.shape()[0]; // number of nodes

	// randomly drop some neurons
	for(int k=0; k<n_nodes; k++){
		if(uni(gen) < p_dropout){
			keep[k] = 0;
		} else {
			keep[k] = 1;
		}
	}

	// cache the input
	last_input = input;
	
	// compute totals
	for(int k=0; k<n_nodes; k++){
		if(keep[k]){
			totals.push_back(biases[k]);
			for(int n=0; n<input.size(); n++){
				totals[k] += weights[k][n] * input[n];
			}
			output.push_back(activation(totals[k]));
		} else {
			totals.push_back(0.);
			output.push_back(0.);
		}
	}
	
	// return the output
	return output;
}
		
vector<double> FullCon::backprop(vector<double> &d_L_d_out, double learn_rate){
	
	// number of nodes and input size
	int n_nodes = d_L_d_out.size();
	int len_input = last_input.size();

	/* Gradients of loss against weights, biases, and input
	   We use that: 
		* d_totals[i]_d_weights[j] = last_input for i = j and 0 otherwise
		* d_totals_d_bias is the identity matrix
		* d_totals_d_inputs = weights                        */				

	d2_array_type d_L_d_w(boost::extents[n_nodes][last_input.size()]);
	vector<double> d_L_d_inputs;
	
	for(int n=0; n<last_input.size(); n++){
		d_L_d_inputs.push_back(0.);
		for(int l=0; l<n_nodes; l++){
			if(keep[l]) {
				// We use that the derivative of the activation function s is s*(1-s)
				d_L_d_w[l][n] = d_L_d_out[l] * activation_p(totals[l]) * last_input[n];
				
				d_L_d_inputs[n] += d_L_d_out[l] * activation_p(totals[l]) * weights[l][n];
	
				// update weights
				weights[l][n] -= learn_rate * d_L_d_w[l][n];
			} else {
				d_L_d_w[l][n] = 0.;
			}
		}
	}

	// update biases
	for(int l=0; l<n_nodes; l++){
		biases[l] -= learn_rate * d_L_d_out[l] * activation_p(totals[l]); 
	}

	// return the gradient of the loss function against the input
	return d_L_d_inputs;
}
