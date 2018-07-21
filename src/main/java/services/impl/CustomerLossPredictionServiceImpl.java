package services.impl;

import org.nd4j.linalg.api.ndarray.INDArray;
import services.CustomerLossPredictionService;

import java.io.File;

public class CustomerLossPredictionServiceImpl implements CustomerLossPredictionService {
    @Override
    public boolean predictCustomerLoss(File savedModel, INDArray data) {
        return false;
    }
}
