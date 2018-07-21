package services;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;

public interface CustomerLossPredictionService {
  public boolean predictCustomerLoss(File savedModel, INDArray data);
}
