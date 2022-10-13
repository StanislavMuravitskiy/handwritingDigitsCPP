#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
using namespace std;


// ������ ���������� ����� � 3 ����� ���������
const int input_nodes = 784;
const int hidden_nodes = 100;
const int output_nodes = 10;

//����������� �������� ���������
const double learning_rate = 0.3;


const double e = 2.71828;


//��������� ������� �� �����
vector<vector<double>> matrixAndNumber(vector<vector<double>> a, double num) {
	for (int i = 0; i < a.size(); i++) {
		for (int j = 0; j < a[0].size(); j++) {
			a[i][j] = a[i][j] * num;
		}
	}
	return a;
}
// ������������� ������� - ��������
vector<vector<double>> sigmoid(vector<vector<double>> a) {
	for (int i = 0; i < a.size(); i++) {
		for (int j = 0; j < a[0].size(); j++) {
			a[i][j] = 1.0 / (1.0 + pow(e, -1 * a[i][j]));
		}
	}
	return a;
}

//��������� �� ��������� �������
vector<vector<double>> matrixSomething(vector<vector<double>> a) {
	for (int i = 0; i < a.size(); i++) {
		for (int j = 0; j < a[0].size(); j++)
			a[i][j] = 1.0 - a[i][j];
	}
	return a;
}

//�������� ������
vector<vector<double>> matrixAddition(vector<vector<double>> a, vector<vector<double>> b) {
	for (int i = 0; i < a.size(); i++) {
		for (int j = 0; j < a[0].size(); j++) {
			a[i][j] += b[i][j];
		}
	}
	return a;
}
//��������� ������
vector<vector<double>> matrixSubstration(vector<vector<double>> a, vector<vector<double>> b) {
	for (int i = 0; i < a.size(); i++) {
		for (int j = 0; j < a[0].size(); j++) {
			a[i][j] -= b[i][j];
		}
	}
	return a;
}

//��������� ������
vector<vector<double>> matrixMultiplication(vector<vector<double>> a, vector<vector<double>> b) {
	vector<vector<double>> c(a.size(), vector<double>(b[0].size(), 0));
	for (int i = 0; i < a.size(); i++) {
		for (int j = 0; j < b[0].size(); j++) {
			for (int k = 0; k < a[0].size(); k++) {
				c[i][j] += a[i][k] * b[k][j];
				
			}
		}
	}
	return c;
	

}
//���������������� �������
vector<vector<double>> matrixTransposition(vector<vector<double>> a) {
	vector<vector<double>> b(a[0].size(), vector<double>(a.size(), 0));
	for (int i = 0; i < a.size(); i++) {
		for (int j = 0; j < a[0].size(); j++) {
			b[j][i] = a[i][j];
		}
	}
	return b;
}
//������������ ��������� ������
vector<vector<double>> matrixMn(vector<vector<double>> a, vector<vector<double>> b) {
	vector<vector<double>> c(a.size(), vector<double>(b[0].size(), 0));
	for (int i = 0; i < a.size(); i++) {
		for (int j = 0; j < b[0].size(); j++) {
			c[i][j] = a[i][j] * b[i][j];
		}
	}
	return c;


}




int main() {

	srand(time(0));

	cout << "1.Training" << endl;
	cout << "2. Test" << endl;
	int n;
	cin >> n;

	switch (n) {
	case 1:
	{

		//������� ������� �����
		vector<vector<double>> input_to_hidden_weights(hidden_nodes, vector<double>(input_nodes, 0));
		vector<vector<double>> hidden_to_output_weights(output_nodes, vector<double>(hidden_nodes, 0));


		// ��������� ������� ����� ���������� ���������� �� 0 �� 1 / ������(hidden_nodes)
		for (int i = 0; i < hidden_nodes; i++) {
			for (int j = 0; j < input_nodes; j++) {
				input_to_hidden_weights[i][j] = double(rand() % 1000) / 10000.0;
			}
		}

		// ����������
		for (int i = 0; i < output_nodes; i++) {
			for (int j = 0; j < hidden_nodes; j++) {
				hidden_to_output_weights[i][j] = double(rand() % 316) / 1000.0;
			}
		}

		// ��������� ������ ��������� �������
		ifstream training_data_input("mnist_train.txt");
		vector<vector<double>> training_data_vector(60000, vector<double>(785, 0));
		for (int i = 0; i < 60000; i++) {
			for (int j = 0; j < 785; j++) {
				training_data_input >> training_data_vector[i][j];
			}
		}

		// �������� �������� � 5 �������
		for (int epochs = 0; epochs < 1; epochs++) {
			for (int i = 0; i < 60000; i++) {
				cout << "Num: " << i << endl;

				// �������� ������� �������� �� 0.1 �� 1
				for (int j = 1; j < 785; j++) {
					training_data_vector[i][j] = training_data_vector[i][j] / 255.0 * 0.99 + 0.1;
				}

				// ������� ������ ����������� � ��������� ���������
				vector<double> results(output_nodes, 0.01);
				int help = training_data_vector[i][0];
				results[help] = 0.99;


				// �������� ������ �������  �������� � ���������� ����
				vector<vector<double>> new_inputs(1, vector<double>(784, 0));
				int k = 0;
				for (int j = 1; j < 785; j++) {
					new_inputs[0][k] = training_data_vector[i][j];
					k++;
				}

				// �������� ������ �������� ����������� � ���������� ����
				vector<vector<double>> new_results(1, vector<double>(results.size(), 0));
				for (int j = 0; j < results.size(); j++) {
					new_results[0][j] = results[j];
				}


				// ������������� ������� ��� ���������� ����������
				new_results = matrixTransposition(new_results);
				new_inputs = matrixTransposition(new_inputs);



				// ������� ������� �������� �������� ����
				vector<vector<double>> hidden_inputs_value = matrixMultiplication(input_to_hidden_weights, new_inputs);

				// ������� �������� �������� �������� ����
				vector<vector<double>> hidden_outputs_value = sigmoid(hidden_inputs_value);


				//������� ������� �������� ��������� ����
				vector<vector<double>> final_inputs_value = matrixMultiplication(hidden_to_output_weights, hidden_outputs_value);

				//������� �������� �������� ��������� ����
				vector<vector<double>> final_outputs_value = sigmoid(final_inputs_value);

				// ������� ������ ��������� ����
				vector<vector<double>> output_errors = matrixSubstration(new_results, final_outputs_value);

				cout << "hello";
				// ������� ������ �������� ����
				vector<vector<double>> hidden_errors = matrixMultiplication(matrixTransposition(hidden_to_output_weights), output_errors);



				//����� ������ ������� ������� ��� ������������ ��������� �����

				// ��� ���
				vector<vector<double>> delta_hidden_to_output_weight =
					matrixMultiplication(matrixMn(matrixMn(output_errors, final_outputs_value), matrixSomething(final_outputs_value)),
						matrixTransposition(hidden_outputs_value));

				// �������� ���� ��������� �� ����������� �������� ��� ����������� �������� � ����� ���������� ����������
				vector<vector<double>> dhtow_with_learning_rate = matrixAndNumber(delta_hidden_to_output_weight, learning_rate);

				// ��������� ���� ��������� � �����
				hidden_to_output_weights = matrixAddition(hidden_to_output_weights, dhtow_with_learning_rate);



				// ����� ��� ����������
				vector<vector<double>> delta_input_to_hidden_weights =
					matrixMultiplication(matrixMn(matrixMn(hidden_errors, hidden_outputs_value), matrixSomething(hidden_outputs_value)),
						matrixTransposition(new_inputs));

				vector<vector<double>> dithw_with_learning_rate = matrixAndNumber(delta_input_to_hidden_weights, learning_rate);

				input_to_hidden_weights = matrixAddition(input_to_hidden_weights, dithw_with_learning_rate);

				// ���!!! �������� �� ����� �� ����� ���������, ��� �� ����� �� ��������� �������






			}
		}

		// ��������� �������� ����� ����� � ��������� ����
		ofstream output("input_to_hidden_weights.txt");
		for (int i = 0; i < hidden_nodes; i++) {
			for (int j = 0; j < input_nodes; j++) {
				output << input_to_hidden_weights[i][j] << " ";
			}
		}
		output.close();
		// ����� �� �� �����
		ofstream output1("hidden_to_output_weights.txt");
		for (int i = 0; i < output_nodes; i++) {
			for (int j = 0; j < hidden_nodes; j++) {
				output1 << hidden_to_output_weights[i][j] << " ";
			}
		}

		output1.close();

		// ��������� �������� �������
		ifstream test_data_input("mnist_test.txt");
		vector<vector<double>> test_data_vector(10000, vector<double>(785, 0));
		for (int i = 0; i < 10000; i++) {
			for (int j = 0; j < 785; j++) {
				test_data_input >> test_data_vector[i][j];
			}
		}


		// ����� ������� ���������� ������
		double right_answers = 0;

		for (int i = 0; i < 10000; i++) {
			// ����� ����������� �����
			for (int j = 1; j < 785; j++) {
				test_data_vector[i][j] = test_data_vector[i][j] / 255.0 * 0.99 + 0.1;
			}
			// ��������� ������� ������ � ��������� ���
			vector<vector<double>> new_inputs(1, vector<double>(784, 0));
			for (int j = 1; j < 785; j++) {
				new_inputs[0][j - 1] = test_data_vector[i][j];
			}
			// ������������� ��� ���������� ����������
			new_inputs = matrixTransposition(new_inputs);

			//�������� ������� ��� ������� ��������
			vector<vector<double>> hidden_inputs_value = matrixMultiplication(input_to_hidden_weights, new_inputs);
			vector<vector<double>> hidden_outputs_value = sigmoid(hidden_inputs_value);

			vector<vector<double>> final_inputs_value = matrixMultiplication(hidden_to_output_weights, hidden_outputs_value);
			vector<vector<double>> final_outputs_value = sigmoid(final_inputs_value);


			// ������� ����� � ���������� ������������
			double max_value = 0;
			int best_result_number = 0;
			for (int j = 0; j < 10; j++) {
				if (final_outputs_value[j][0] > max_value) {
					max_value = final_outputs_value[j][0];
					best_result_number = j;

				}
			}

			// ���� ��� ���������� ����������� �������
			if (best_result_number == test_data_vector[i][0]) {
				right_answers++;
			}



		}

		// ������� �������� ��������� �� �������� �������
		cout << "Training and tests completed" << endl;
		cout << "Accuracy: " << right_answers / 100 << endl;



		break;
	}

	case 2:
	{

		//���������� ��������� ��� ����� ������

		//������� ������� �����
		vector<vector<double>> new_input_weights(hidden_nodes, vector<double>(input_nodes, 0));
		vector<vector<double>> new_output_weights(output_nodes, vector<double>(hidden_nodes, 0));



		// ��������� ��� ����, ������� ��������� �����
		ifstream input("input_to_hidden_weights.txt");
		for (int i = 0; i < hidden_nodes; i++) {
			for (int j = 0; j < input_nodes; j++) {
				input >> new_input_weights[i][j];
			}
		}
		input.close();

		ifstream input1("hidden_to_output_weights.txt");
		for (int i = 0; i < output_nodes; i++) {
			for (int j = 0; j < hidden_nodes; j++) {
				input1 >> new_output_weights[i][j];
			}
		}
		input1.close();


		// ��������� ��������� ������������� ����� �������� � �����������
		ifstream input2("img.txt");
		vector<double> img_vector(784);

		for (int i = 0; i < 784; i++) {
			input2 >> img_vector[i];
			img_vector[i] = img_vector[i] / 255.0 * 0.99 + 0.01;
		}
		input2.close();


		// �������� � ���������� ����
		vector<vector<double>> new_inputs(1, vector<double>(784, 0));

		for (int j = 0; j < 784; j++) {
			new_inputs[0][j] = img_vector[j];
		}

		// ����� �������������
		new_inputs = matrixTransposition(new_inputs);

		//����������� ��� �����������
		vector<vector<double>> hidden_inputs_value = matrixMultiplication(new_input_weights, new_inputs);
		vector<vector<double>> hidden_outputs_value = sigmoid(hidden_inputs_value);

		vector<vector<double>> final_inputs_value = matrixMultiplication(new_output_weights, hidden_outputs_value);
		vector<vector<double>> final_outputs_value = sigmoid(final_inputs_value);


		//������� ������� ������������ ��������
		for (int i = 0; i < 10; i++) {
			cout << i << ": " << final_outputs_value[i][0] << endl;
		}


		//�����. ������ - ����� � ����� - ����!!!








		break;
	}

	default:
		break;
	}
	return 0;
}