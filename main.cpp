#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <cmath>

using namespace std;


// Задаем количество узлов в 3 слоях нейросети
const int input_nodes = 784;
const int hidden_nodes = 100;
const int output_nodes = 10;

//Коэффициент обучения нейросети
const double learning_rate = 0.3;


const double e = 2.71828;


//Умножение матрицы на число
vector<vector<double>> matrixAndNumber(vector<vector<double>> a, double num) {
	for (int i = 0; i < a.size(); i++) {
		for (int j = 0; j < a[0].size(); j++) {
			a[i][j] = a[i][j] * num;
		}
	}
	return a;
}
// Активационная функция - сигмоида
vector<vector<double>> sigmoid(vector<vector<double>> a) {
	for (int i = 0; i < a.size(); i++) {
		for (int j = 0; j < a[0].size(); j++) {
			a[i][j] = 1.0 / (1.0 + pow(e, -1 * a[i][j]));
		}
	}
	return a;
}

//Вычитание из единичной матрицы
vector<vector<double>> matrixSomething(vector<vector<double>> a) {
	for (int i = 0; i < a.size(); i++) {
		for (int j = 0; j < a[0].size(); j++)
			a[i][j] = 1.0 - a[i][j];
	}
	return a;
}

//Сложение матриц
vector<vector<double>> matrixAddition(vector<vector<double>> a, vector<vector<double>> b) {
	for (int i = 0; i < a.size(); i++) {
		for (int j = 0; j < a[0].size(); j++) {
			a[i][j] += b[i][j];
		}
	}
	return a;
}
//Вычитание матриц
vector<vector<double>> matrixSubstration(vector<vector<double>> a, vector<vector<double>> b) {
	for (int i = 0; i < a.size(); i++) {
		for (int j = 0; j < a[0].size(); j++) {
			a[i][j] -= b[i][j];
		}
	}
	return a;
}

//Умножение матриц
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
//Транспонирование матрицы
vector<vector<double>> matrixTransposition(vector<vector<double>> a) {
	vector<vector<double>> b(a[0].size(), vector<double>(a.size(), 0));
	for (int i = 0; i < a.size(); i++) {
		for (int j = 0; j < a[0].size(); j++) {
			b[j][i] = a[i][j];
		}
	}
	return b;
}
//Поэлементное умножение матриц
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

		//Создаем матрицы весов
		vector<vector<double>> input_to_hidden_weights(hidden_nodes, vector<double>(input_nodes, 0));
		vector<vector<double>> hidden_to_output_weights(output_nodes, vector<double>(hidden_nodes, 0));


		// Заполняем матрицы весов случайными значениями от 0 до 1 / корень(hidden_nodes)
		for (int i = 0; i < hidden_nodes; i++) {
			for (int j = 0; j < input_nodes; j++) {
				input_to_hidden_weights[i][j] = double(rand() % 1000) / 10000.0;
			}
		}

		// Аналогично
		for (int i = 0; i < output_nodes; i++) {
			for (int j = 0; j < hidden_nodes; j++) {
				hidden_to_output_weights[i][j] = double(rand() % 316) / 1000.0;
			}
		}

		// Считываем данные обучающей выборки
		ifstream training_data_input("mnist_train.txt");
		vector<vector<double>> training_data_vector(60000, vector<double>(785, 0));
		for (int i = 0; i < 60000; i++) {
			for (int j = 0; j < 785; j++) {
				training_data_input >> training_data_vector[i][j];
			}
		}

		// Начинаем обучение с 1 эпохой
		for (int epochs = 0; epochs < 1; epochs++) {
		    cout << endl;
			for (int i = 0; i < 60000; i++) {
				if(i%100==0){
				    cout << "\x1b[1A" << "\x1b[2K"; // Delete the entire line
				    cout << "Num: " << i << endl;
				}
				// Приводим входные значения от 0.1 до 1
				for (int j = 1; j < 785; j++) {
					training_data_vector[i][j] = training_data_vector[i][j] / 255.0 * 0.99 + 0.1;
				}

				// Создаем вектор результатов с ожидаемым значением
				vector<double> results(output_nodes, 0.01);
				int help = training_data_vector[i][0];
				results[help] = 0.99;


				// Приводим вектор входных  значений к матричному виду
				vector<vector<double>> new_inputs(1, vector<double>(784, 0));
				int k = 0;
				for (int j = 1; j < 785; j++) {
					new_inputs[0][k] = training_data_vector[i][j];
					k++;
				}

				// Приводим вектор выходных результатов к матричному виду
				vector<vector<double>> new_results(1, vector<double>(results.size(), 0));
				for (int j = 0; j < results.size(); j++) {
					new_results[0][j] = results[j];
				}


				// Транспонируем матрицы для дальнейших вычисление
				new_results = matrixTransposition(new_results);
				new_inputs = matrixTransposition(new_inputs);



				// Находим входные значения скрытого слоя
				vector<vector<double>> hidden_inputs_value = matrixMultiplication(input_to_hidden_weights, new_inputs);

				// Находим выходные значения скрытого слоя
				vector<vector<double>> hidden_outputs_value = sigmoid(hidden_inputs_value);


				//Находим входные значения выходного слоя
				vector<vector<double>> final_inputs_value = matrixMultiplication(hidden_to_output_weights, hidden_outputs_value);

				//Находим выходные значения выходного слоя
				vector<vector<double>> final_outputs_value = sigmoid(final_inputs_value);

				// Находим ошибки выходного слоя
				vector<vector<double>> output_errors = matrixSubstration(new_results, final_outputs_value);

				// cout << "hello";
				// Находим ошибки скрытого слоя
				vector<vector<double>> hidden_errors = matrixMultiplication(matrixTransposition(hidden_to_output_weights), output_errors);



				//Здесь дальше сложная формула для высчитывания изменения весов

				// Это она
				vector<vector<double>> delta_hidden_to_output_weight =
					matrixMultiplication(matrixMn(matrixMn(output_errors, final_outputs_value), matrixSomething(final_outputs_value)),
						matrixTransposition(hidden_outputs_value));

				// Умножаем наше изменение на коэффициент обучения для дальнейшего обучения и учета предыдущих достижений
				vector<vector<double>> dhtow_with_learning_rate = matrixAndNumber(delta_hidden_to_output_weight, learning_rate);

				// Добавляем наши изменения к весам
				hidden_to_output_weights = matrixAddition(hidden_to_output_weights, dhtow_with_learning_rate);



				// Здесь все анологично
				vector<vector<double>> delta_input_to_hidden_weights =
					matrixMultiplication(matrixMn(matrixMn(hidden_errors, hidden_outputs_value), matrixSomething(hidden_outputs_value)),
						matrixTransposition(new_inputs));

				vector<vector<double>> dithw_with_learning_rate = matrixAndNumber(delta_input_to_hidden_weights, learning_rate);

				input_to_hidden_weights = matrixAddition(input_to_hidden_weights, dithw_with_learning_rate);

				// Ура!!! Обучение на одном из чисел закончено, так со всеми из обучающей выборки






			}
		}

		// Сохраняем значения наших весов в текстовый файл
		ofstream output("input_to_hidden_weights.txt");
		for (int i = 0; i < hidden_nodes; i++) {
			for (int j = 0; j < input_nodes; j++) {
				output << input_to_hidden_weights[i][j] << " ";
			}
		}
		output.close();
		// Здесь то же самое
		ofstream output1("hidden_to_output_weights.txt");
		for (int i = 0; i < output_nodes; i++) {
			for (int j = 0; j < hidden_nodes; j++) {
				output1 << hidden_to_output_weights[i][j] << " ";
			}
		}

		output1.close();

		// Считываем тестовую выборку
		ifstream test_data_input("mnist_test.txt");
		vector<vector<double>> test_data_vector(10000, vector<double>(785, 0));
		for (int i = 0; i < 10000; i++) {
			for (int j = 0; j < 785; j++) {
				test_data_input >> test_data_vector[i][j];
			}
		}


		// Будем считать правильные ответы
		double right_answers = 0;

		for (int i = 0; i < 10000; i++) {
			// Снова преобразуем числа
			for (int j = 1; j < 785; j++) {
				test_data_vector[i][j] = test_data_vector[i][j] / 255.0 * 0.99 + 0.1;
			}
			// Переводим входные данные в матричный вид
			vector<vector<double>> new_inputs(1, vector<double>(784, 0));
			for (int j = 1; j < 785; j++) {
				new_inputs[0][j - 1] = test_data_vector[i][j];
			}
			// Транспонируем для дальнейших вычислений
			new_inputs = matrixTransposition(new_inputs);

			//Поэтапно считаем все входные значения
			vector<vector<double>> hidden_inputs_value = matrixMultiplication(input_to_hidden_weights, new_inputs);
			vector<vector<double>> hidden_outputs_value = sigmoid(hidden_inputs_value);

			vector<vector<double>> final_inputs_value = matrixMultiplication(hidden_to_output_weights, hidden_outputs_value);
			vector<vector<double>> final_outputs_value = sigmoid(final_inputs_value);


			// Находим цифру с наибольшей вероятностью
			double max_value = 0;
			int best_result_number = 0;
			for (int j = 0; j < 10; j++) {
				if (final_outputs_value[j][0] > max_value) {
					max_value = final_outputs_value[j][0];
					best_result_number = j;

				}
			}

			// Если она правильная увеличиваем счетчик
			if (best_result_number == test_data_vector[i][0]) {
				right_answers++;
			}



		}

		// Выводим точность нейросети на тестовой выборке
		cout << "Training and tests completed" << endl;
		cout << "Accuracy: " << right_answers / 100 << endl;



		break;
	}

	case 2:
	{

		//Используем нейросеть для наших данных

		//Создаем матрицы весов
		vector<vector<double>> new_input_weights(hidden_nodes, vector<double>(input_nodes, 0));
		vector<vector<double>> new_output_weights(output_nodes, vector<double>(hidden_nodes, 0));



		// Считываем все веса, которые сохранили ранее
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


		// Считываем текстовое представление нашей картинки и преобразуем
		ifstream input2("img.txt");
		vector<double> img_vector(784);

		for (int i = 0; i < 784; i++) {
			input2 >> img_vector[i];
			img_vector[i] = img_vector[i] / 255.0 * 0.99 + 0.01;
		}
		input2.close();


		// Приводим к матричному виду
		vector<vector<double>> new_inputs(1, vector<double>(784, 0));

		for (int j = 0; j < 784; j++) {
			new_inputs[0][j] = img_vector[j];
		}

		// Снова транспонируем
		new_inputs = matrixTransposition(new_inputs);

		//Потихонечку все высчитываем
		vector<vector<double>> hidden_inputs_value = matrixMultiplication(new_input_weights, new_inputs);
		vector<vector<double>> hidden_outputs_value = sigmoid(hidden_inputs_value);

		vector<vector<double>> final_inputs_value = matrixMultiplication(new_output_weights, hidden_outputs_value);
		vector<vector<double>> final_outputs_value = sigmoid(final_inputs_value);


		//Выводим таблицу вероятностей значений
		for (int i = 0; i < 10; i++) {
			cout << i << ": " << final_outputs_value[i][0] << endl;
		}


		//Конец. Выводы - линал и матан - сила!!!








		break;
	}

	default:
		break;
	}
	return 0;
}
