void ReLU_v::save(ofstream& file){
	save_vector(last_input, file);
}

void ReLU_v::load(ifstream& file){
	vector<double> last_input_ = load_vector<double>(file);
	last_input = last_input_;
}
		
vector<double> ReLU_v::forward(vector<double> &input){

	int size = input.size();
	
	// cache the input
	last_input = input;
	
	// initialize and compute the output
	vector<double> output;
	for(int n=0; n<size; n++){
		double el = input[n];
		if(el >= 0.){
			output.push_back(el);
		} else {
			output.push_back(0.);
		}
	}
	return output;
}
		
vector<double> ReLU_v::backprop(vector<double> &d_L_d_out){
	
	int size = last_input.size(); 

	// gradient of the loss function against the input
	vector<double> d_L_d_input;
	for(int n=0; n<size; n++){
		double el = last_input[n];
		if(el >= 0.){
			d_L_d_input.push_back(d_L_d_out[n]);
		} else {
			d_L_d_input.push_back(0.);
		}
	}
	return d_L_d_input;
}
