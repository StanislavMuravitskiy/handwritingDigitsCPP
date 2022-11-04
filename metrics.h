double accuracy_score(Matrix y_true, Matrix y_test){
		int tp = 0; // True positive
		int fp = 0; // False positive
		int tn = 0; // True negative
		int fn = 0; // False negative
		for(int i = 0; i < y_true.width(); i++)
		    for(int j = 0; j < y_true.height(); j++){
				if(y_test.getItem(i,j)>=0.5 && y_true.getItem(i,j)>=0.5)
				    tp++;
				else if(y_test.getItem(i,j)>=0.5 && y_true.getItem(i,j)<0.5)
				    fp++;
				else if(y_test.getItem(i,j)<0.5 && y_true.getItem(i,j)<0.5)
					tn++;
		        else if(y_test.getItem(i,j)<0.5 && y_true.getItem(i,j)>=0.5)
					fn++;
				}
		return (tp+tn)/(tp+tn+fp+fn);
}
