def predict_class(model, instance, num_classes):
    '''
        Uses a trained ranking model to classify a given instance vector.
        The model takes inputs
            X: instance
            Y: one-of-K encoded class vector
        
        returns a class
    '''
    import numpy as np
    scores = list()
    all_classes = np.eye(num_classes)
    inst = np.reshape(instance,(1,28,28,1))
    for i in range(len(all_classes)):
        cl = all_classes[i]
        cl = np.reshape(cl,(1,num_classes))
        scores.append(model.predict([inst, cl]))
    
    sas = np.squeeze(np.asarray(scores))
    ordering = sas.argsort()[::-1][:num_classes]+1
    return ordering[0], scores