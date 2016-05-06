import numpy as np
from copy import deepcopy
# from Optimization import minimize, scg

class Optimizer(object):
    def __init__(self, model=None, searchConfig = None):
        self.model = model

    def _convert_to_array(self):
        '''Convert all hyparameters in the model to an array'''
        hyplist = self.model.meanfunc.hyp + self.model.covfunc.hyp + self.model.likfunc.hyp
        return np.array(hyplist)

    def _nlzAnddnlz(self, hypInArray):
        '''Find negative-log-marginal-likelihood and derivatives in one pass(faster)'''
        self._apply_in_objects(hypInArray)
        nlZ, dnlZ, post = self.model.getPosterior()
        dnlml_List = dnlZ.mean + dnlZ.cov + dnlZ.lik
        return nlZ, np.array(dnlml_List)

class Minimize(Optimizer):
    '''minimize by Carl Rasmussen (python implementation of "minimize" in GPML)'''
    def __init__(self, model, searchConfig = None):
        super(Minimize, self).__init__()
        self.model = model
        self.searchConfig = searchConfig
        self.trailsCounter = 0
        self.errorCounter = 0

    def findMin(self, x, y, numIters = 100):
        meanfunc = self.model.meanfunc
        covfunc = self.model.covfunc
        likfunc = self.model.likfunc
        inffunc = self.model.inffunc
        hypInArray = self._convert_to_array()

        try:
            opt = minimize.run(self._nlzAnddnlz, hypInArray, length=-numIters)
            optimalHyp = deepcopy(opt[0])
            funcValue  = opt[1][-1]
        except:
            self.errorCounter += 1
            if not self.searchConfig:
                raise Exception("Can not learn hyperparamters using minimize.")
        self.trailsCounter += 1

        if self.searchConfig:
            searchRange = self.searchConfig.meanRange + self.searchConfig.covRange + self.searchConfig.likRange
            if not (self.searchConfig.num_restarts or self.searchConfig.min_threshold):
                raise Exception('Specify at least one of the stop conditions')
            while True:
                self.trailsCounter += 1                 # increase counter
                for i in range(hypInArray.shape[0]):   # random init of hyp
                    hypInArray[i]= np.random.uniform(low=searchRange[i][0], high=searchRange[i][1])
                # value this time is better than optiaml min value
                try:
                    thisopt = minimize.run(self._nlzAnddnlz, hypInArray, length=-40)
                    if thisopt[1][-1] < funcValue:
                        funcValue  = thisopt[1][-1]
                        optimalHyp = thisopt[0]
                except:
                    self.errorCounter += 1
                if self.searchConfig.num_restarts and self.errorCounter > old_div(self.searchConfig.num_restarts,2):
                    print("[Minimize] %d out of %d trails failed during optimization" % (self.errorCounter, self.trailsCounter))
                    raise Exception("Over half of the trails failed for minimize")
                if self.searchConfig.num_restarts and self.trailsCounter > self.searchConfig.num_restarts-1:         # if exceed num_restarts
                    print("[Minimize] %d out of %d trails failed during optimization" % (self.errorCounter, self.trailsCounter))
                    return optimalHyp, funcValue
                if self.searchConfig.min_threshold and funcValue <= self.searchConfig.min_threshold:           # reach provided mininal
                    print("[Minimize] %d out of %d trails failed during optimization" % (self.errorCounter, self.trailsCounter))
                    return optimalHyp, funcValue
        return optimalHyp, funcValue
