#include <vector>

using namespace std;

const double e = 2.71828;

class Matrix{
    public:
        Matrix(int width, int height, double var){
		    vector<double> line(height, var);
            vector<vector<double>> buf(width, line);
		    matrix = buf;
        }

        //Умножение матрицы на число
        Matrix matrixAndNumber(double num) {
            Matrix result(this->width(), this->height(), 0);
            for (int i = 0; i < matrix.size(); i++) {
                for (int j = 0; j < matrix[0].size(); j++) {
                    result.matrix[i][j] = matrix[i][j] * num;
                }
            }
            return result;
        }

        //Вычитание из единичной матрицы
        Matrix negative(){
            Matrix result(this->width(), this->height(), 0);
            for (int i = 0; i < matrix.size(); i++) {
                for (int j = 0; j < matrix[0].size(); j++)
                    result.matrix[i][j] = 1.0 - matrix[i][j];
            }
            return result;
        }
            
        //Сложение матриц
        Matrix addition(Matrix b) {
            Matrix result(this->width(), this->height(), 0);
            for (int i = 0; i < matrix.size(); i++) {
                for (int j = 0; j < matrix[0].size(); j++) {
                    result.matrix[i][j] = matrix[i][j]+b.matrix[i][j];
                }
            }
            return result;
        }

        //Вычитание матриц
        Matrix substration(Matrix b) {
            Matrix result(this->width(), this->height(), 0);
            for (int i = 0; i < matrix.size(); i++) {
                for (int j = 0; j < matrix[0].size(); j++) {
                    result.matrix[i][j] = matrix[i][j]-b.matrix[i][j];
                }
            }
            return result;
        }
        
        //Умножение матриц
        Matrix dot(Matrix b) {
            Matrix result(this->width(), b.height(), 0);
            for (int i = 0; i < this->width(); i++) {
                for (int j = 0; j < b.height(); j++) {
                    for (int k = 0; k < matrix[0].size(); k++) {
                        result.matrix[i][j] += matrix[i][k] * b.matrix[k][j];                
                    }
                }
            }
            return result;
        }
        
		//Транспонирование матрицы
        Matrix transpose() {
            Matrix result(this->height(), this->width(), 0);
            for (int i = 0; i < this->width(); i++) {
                for (int j = 0; j < this->height(); j++) {
                    result.matrix[j][i] = matrix[i][j];
                }
            }
            return result;
        }
        
		//Поэлементное умножение матриц
        Matrix multiply(Matrix b) {
            Matrix result(this->width(), this->height(), 0);
            for (int i = 0; i < matrix.size(); i++) {
                for (int j = 0; j < b.height(); j++) {
                    result.matrix[i][j] = matrix[i][j] * b.matrix[i][j];
                }
            }
            return result;
        }
        
		//Возведение матрицы в квадрат
        Matrix sqr() {
            Matrix result(this->width(), this->height(), 0);
            for (int i = 0; i < matrix.size(); i++) {
                for (int j = 0; j < matrix[0].size(); j++) {
                    result.matrix[i][j] = pow(matrix[i][j],2);
                }
            }
            return result;
        }

		// Активационная функция - сигмоида
        Matrix sigmoid() {
            Matrix result(this->width(), this->height(), 0);
            for (int i = 0; i < matrix.size(); i++) {
                for (int j = 0; j < matrix[0].size(); j++) {
                    result.matrix[i][j] = 1.0 / (1.0 + pow(e, -1 * matrix[i][j]));
                }
            }
            return result;
        }
		
		// Размер матрицы
		int width() {
			return matrix.size();
		}

		int height() {
			return matrix[0].size();
		}

		double getItem(int i, int j){
		    return matrix[i][j];
		}

		void setItem(double var, int i, int j){
		    matrix[i][j] = var;
		}
	private:
        vector<vector<double>> matrix;

};
