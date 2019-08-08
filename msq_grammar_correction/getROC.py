import matplotlib.pyplot as plt
import numpy as np 
def getROCcurve(output, true_values):
    
    output = np.array(output)
    true_values = np.array(true_values)
    #I initialise matrices to store the values for each threshold 
    true_pos= np.zeros((101,1))
    false_neg = np.zeros((101,1))
    true_neg = np.zeros((101,1))
    false_pos = np.zeros((101,1))
    recall = np.zeros((101,1))
    false_pos_rate = np.zeros((101,1))
    precision = np.zeros((101,1))
    f_scores = np.zeros((101,1))
    for T in range(101): 
        
        output_copy = np.matrix.flatten(output) 
        output_copy[output_copy >= (T/100)] = 1 
        output_copy[output_copy != 1] = 0 
        
        true1s = np.argwhere(true_values == 1) 
        true0s = np.argwhere(true_values == 0)
        true1s = np.matrix.flatten(true1s) 
        true0s = np.matrix.flatten(true0s)
        
        predicted_1s = output_copy[true1s] #finds the values where there should be a 1
        predicted_0s = output_copy[true0s] #finds the values where there should be a 0
        
        true_pos[T] = len(predicted_1s[predicted_1s== 1]) #counts true positives 
        true_neg[T] = len(predicted_0s[predicted_0s == 0]) #counts true negatives 
        
        false_pos[T] = len(predicted_0s[predicted_0s == 1]) #counts false positives 
        false_neg[T] = len(predicted_1s[predicted_1s == 0]) #counts false negatives 
        
        #Now, I can calculte the recall (true positive rate) and false positive rate 
        #for the given theshold.
        pos = true_pos[T] + false_neg[T]
        neg = false_pos[T] + true_neg[T]
    
        if pos == 0: #avoids division by 0 
            recall[T] = 0
        else:
            recall[T] = true_pos[T]/pos
            
        if neg == 0: #avoids division by 0 
            false_pos_rate[T] = 0
        else:
            false_pos_rate[T] = false_pos[T]/neg
         
        precision[T] = true_pos[T]/(true_pos[T]+false_pos[T])
        f_scores[T] = ((1.0 + 0.5*0.5) * precision[T] * recall[T]/ ((0.5*0.5 * precision[T]) + recall[T])) if (precision[T]+recall[T] > 0.0) else 0.0
   
    best_threshold = np.argmax(f_scores)
    best_val = np.amax(f_scores)
    #Now I will plot the ROC curve.
    #x = np.linspace(0,1,100)
    #plt.figure()
    #plt.plot(false_pos_rate,recall)
    #plt.plot(x,x)
    #plt.xlabel('False postive rate')
    #plt.ylabel('True positive rate')
    #plt.show()
    #area_under_curve = np.trapz(np.sort(np.squeeze(recall)), np.sort(np.squeeze(false_pos_rate)))
    #print('The area under the curve is {}'.format(area_under_curve))
    print('Best threshold:{}, best f_05 score:{}'.format(best_threshold, best_val))
    
    return best_threshold, best_val