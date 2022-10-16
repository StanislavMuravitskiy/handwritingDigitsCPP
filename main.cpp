#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <cmath>
#include "matrix.h"

using namespace std;

// Задаем количество узлов в 3 слоях нейросети
const int input_nodes = 784;
const int hidden_nodes = 100;
const int output_nodes = 10;

//Коэффициент обучения нейросети
const double learning_rate = 0.3;

double accuracy_score(Matrix y_true, Matrix y_test){
		int tp = 0; // True positive
		int fp = 0; // False positive
		int tn = 0; // True negative
		int fn = 0; // False negative
		for(int i = 0; i < y_true.matrix.size(); i++)
		    for(int j = 0; j < y_true.matrix[0].size(); j++){
				if(y_test.matrix[i][j]>=0.5 && y_true.matrix[i][j]>=0.5)
				    tp++;
				else if(y_test.matrix[i][j]>=0.5 && y_true.matrix[i][j]<0.5)
				    fp++;
				else if(y_test.matrix[i][j]<0.5 && y_true.matrix[i][j]<0.5)
					tn++;
		        else if(y_test.matrix[i][j]<0.5 && y_true.matrix[i][j]>=0.5)
					fn++;
				}
		return (tp+tn)/(tp+tn+fp+fn);
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
        Matrix input_to_hidden_weights(hidden_nodes, vector<double>(input_nodes, 0));
        Matrix hidden_to_output_weights(output_nodes, vector<double>(hidden_nodes, 0));

        //Заполняем матрицы весов случайными значениями от 0 до 1 / корень(hidden_nodes)
        for (int i = 0; i < hidden_nodes; i++) {
            for (int j = 0; j < input_nodes; j++) {
				double val = double(rand() % 1000) / 10000.0;
                input_to_hidden_weights.matrix[i][j] = val;
            }
        }

        //Аналогично
        for (int i = 0; i < output_nodes; i++) {
            for (int j = 0; j < hidden_nodes; j++) {
                hidden_to_output_weights.matrix[i][j] = double(rand() % 316) / 1000.0;
            }
        }

        //Считываем данные обучающей выборки
        ifstream training_data_input("mnist_train.txt");
        Matrix training_data_vector(60000, vector<double>(785, 0));
        for (int i = 0; i < 60000; i++) {
            for (int j = 0; j < 785; j++) {
                training_data_input >> training_data_vector.matrix[i][j];
            }
        }

        //Начинаем обучение с 1 эпохой
        for (int epochs = 0; epochs < 1; epochs++) {
            cout << endl;
            for (int i = 0; i < 60000; i++) {
                if(i%100==0){
                    cout << "\x1b[1A" << "\x1b[2K"; // Delete the entire line
                    cout << "Num: " << i << endl;
                }
                // Приводим входные значения от 0.1 до 1
                for (int j = 1; j < 785; j++) {
                    training_data_vector.matrix[i][j] = training_data_vector.matrix[i][j] / 255.0 * 0.99 + 0.1;
                }

                // Создаем вектор результатов с ожидаемым значением
                vector<double> results(output_nodes, 0.01);
                int help = training_data_vector.matrix[i][0];
                results[help] = 0.99;


                // Приводим вектор входных  значений к матричному виду
                Matrix new_inputs(1, vector<double>(784, 0));
                int k = 0;
                for (int j = 1; j < 785; j++) {
                    new_inputs.matrix[0][k] = training_data_vector.matrix[i][j];
                    k++;
                }

                // Приводим вектор выходных результатов к матричному виду
                Matrix new_results(1, vector<double>(results.size(), 0));
                for (int j = 0; j < results.size(); j++) {
                    new_results.matrix[0][j] = results[j];
                }


                // Транспонируем матрицы для дальнейших вычисление
                new_results = new_results.transpose();
                new_inputs = new_inputs.transpose();



                // Находим входные значения скрытого слоя
                Matrix hidden_inputs_value = input_to_hidden_weights.dot(new_inputs);

                // Находим выходные значения скрытого слоя
                Matrix hidden_outputs_value = hidden_inputs_value.sigmoid();


                //Находим входные значения выходного слоя
                Matrix final_inputs_value = hidden_to_output_weights.dot(hidden_outputs_value);

                //Находим выходные значения выходного слоя
                Matrix final_outputs_value = final_inputs_value.sigmoid();

                // Находим ошибки выходного слоя
                Matrix output_errors = new_results.substration(final_outputs_value);

                // cout << "hello";
                // Находим ошибки скрытого слоя
                Matrix hidden_errors = hidden_to_output_weights.transpose().dot(output_errors);



                //Здесь дальше сложная формула для высчитывания изменения весов

                // Это она
                Matrix delta_hidden_to_output_weight =
                    output_errors.multiply(final_outputs_value).multiply(final_outputs_value.negative()).dot(hidden_outputs_value.transpose());

                // Умножаем наше изменение на коэффициент обучения для дальнейшего обучения и учета предыдущих достижений
                Matrix dhtow_with_learning_rate = delta_hidden_to_output_weight.matrixAndNumber(learning_rate);

                // Добавляем наши изменения к весам
                hidden_to_output_weights = hidden_to_output_weights.addition(dhtow_with_learning_rate);

                // Здесь все анологично
                Matrix delta_input_to_hidden_weights = hidden_errors.multiply(hidden_outputs_value).multiply(hidden_outputs_value.negative()).dot(new_inputs.transpose());
                Matrix dithw_with_learning_rate = delta_input_to_hidden_weights.matrixAndNumber(learning_rate);
                input_to_hidden_weights = input_to_hidden_weights.addition(dithw_with_learning_rate);

                // Ура!!! Обучение на одном из чисел закончено, так со всеми из обучающей выборки

            }
        }

        // Сохраняем значения наших весов в текстовый файл
        ofstream output("input_to_hidden_weights.txt");
        for (int i = 0; i < hidden_nodes; i++) {
            for (int j = 0; j < input_nodes; j++) {
                output << input_to_hidden_weights.matrix[i][j] << " ";
            }
        }
        output.close();
        // Здесь то же самое
        ofstream output1("hidden_to_output_weights.txt");
        for (int i = 0; i < output_nodes; i++) {
            for (int j = 0; j < hidden_nodes; j++) {
                output1 << hidden_to_output_weights.matrix[i][j] << " ";
            }
        }

        output1.close();

        // Считываем тестовую выборку
        ifstream test_data_input("mnist_test.txt");
        Matrix test_data_vector(10000, vector<double>(785, 0));
        for (int i = 0; i < 10000; i++) {
            for (int j = 0; j < 785; j++) {
                test_data_input >> test_data_vector.matrix[i][j];
            }
        }


        // Будем считать правильные ответы
        double right_answers = 0;
		double accuracy = 0;

        for (int i = 0; i < 10000; i++) {
            // Снова преобразуем числа
            for (int j = 1; j < 785; j++) {
                test_data_vector.matrix[i][j] = test_data_vector.matrix[i][j] / 255.0 * 0.99 + 0.1;
            }
            // Переводим входные данные в матричный вид
            Matrix new_inputs(1, vector<double>(784, 0));
            for (int j = 1; j < 785; j++) {
                new_inputs.matrix[0][j - 1] = test_data_vector.matrix[i][j];
            }

            //Поэтапно считаем все входные значения
            Matrix hidden_inputs_value = input_to_hidden_weights.dot(new_inputs.transpose());
            Matrix hidden_outputs_value = hidden_inputs_value.sigmoid();

            Matrix final_inputs_value = hidden_to_output_weights.dot(hidden_outputs_value);
            Matrix final_outputs_value = final_inputs_value.sigmoid();

            // Создаем вектор результатов с ожидаемым значением
            Matrix true_results(1, vector<double>(output_nodes, 0));
            int help = test_data_vector.matrix[i][0];
            true_results.matrix[0][help] = 0.99;
            
		    accuracy+=accuracy_score(true_results, final_outputs_value.transpose());
        }

        // Выводим точность нейросети на тестовой выборке
        cout << "Training and tests completed" << endl;
		cout << "Accuracy: " << accuracy /100 << " %" << endl;

        break;
    }

    case 2:
    {

        //Используем нейросеть для наших данных

        //Создаем матрицы весов
        Matrix new_input_weights(hidden_nodes, vector<double>(input_nodes, 0));
        Matrix new_output_weights(output_nodes, vector<double>(hidden_nodes, 0));



        // Считываем все веса, которые сохранили ранее
        ifstream input("input_to_hidden_weights.txt");
        for (int i = 0; i < hidden_nodes; i++) {
            for (int j = 0; j < input_nodes; j++) {
                input >> new_input_weights.matrix[i][j];
            }
        }
        input.close();

        ifstream input1("hidden_to_output_weights.txt");
        for (int i = 0; i < output_nodes; i++) {
            for (int j = 0; j < hidden_nodes; j++) {
                input1 >> new_output_weights.matrix[i][j];
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
        Matrix new_inputs(1, vector<double>(784, 0));

        for (int j = 0; j < 784; j++) {
            new_inputs.matrix[0][j] = img_vector[j];
        }

        // Снова транспонируем
        new_inputs = new_inputs.transpose();

        //Потихонечку все высчитываем
        Matrix hidden_inputs_value = new_input_weights.dot(new_inputs);
        Matrix hidden_outputs_value = hidden_inputs_value.sigmoid();

        Matrix final_inputs_value = new_output_weights.dot(hidden_outputs_value);
        Matrix final_outputs_value = final_inputs_value.sigmoid();


        //Выводим таблицу вероятностей значений
        for (int i = 0; i < 10; i++) {
            cout << i << ": " << final_outputs_value.matrix[i][0] << endl;
        }


        //Конец. Выводы - линал и матан - сила!!!

        break;
    }

    default:
        break;
    }
    return 0;
}
