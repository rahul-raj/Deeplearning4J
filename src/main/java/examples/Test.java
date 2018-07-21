package examples;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.KFoldIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.indexing.conditions.AbsValueGreaterThan;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Test {

    private static Logger log = LoggerFactory.getLogger("examples.CustomerLossPrediction.class");
    public static void main(String[] args) throws IOException, InterruptedException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {

        Schema schema = new Schema.Builder()
                .addColumnsString("RowNumber")
                .addColumnInteger("CustomerId")
                .addColumnString("Surname")
                .addColumnInteger("CreditScore")
                .addColumnCategorical("Geography",Arrays.asList("France","Spain","Germany"))
                .addColumnCategorical("Gender",Arrays.asList("Male","Female"))
                .addColumnsInteger("Age","Tenure","Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary","Exited").build();
        TransformProcess transformProcess = new TransformProcess.Builder(schema)
                .removeColumns("RowNumber","Surname","CustomerId")
                .categoricalToInteger("Gender")
                .categoricalToOneHot("Geography")
                .removeColumns("Geography[France]")
                .build();
        RecordReader reader = new CSVRecordReader(1,',');
        reader.initialize(new FileSplit(new ClassPathResource("Churn_Modelling.csv").getFile()));
        RecordReader transformProcessRecordReader = new TransformProcessRecordReader(reader,transformProcess);

        int labelIndex = 11;
        int numClasses = 1;
        int batchSize = 10;

        DataSetIterator iterator = new RecordReaderDataSetIterator(transformProcessRecordReader,batchSize,labelIndex,numClasses);
        List<DataSet> dataList = new ArrayList<>();

        while(iterator.hasNext()){
            dataList.add(iterator.next());
        }

        DataSet allData = DataSet.merge(dataList);
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.80);

        DataSet trainSet = testAndTrain.getTrain();
        DataSet testSet = testAndTrain.getTest();

        //System.out.println("args = [" + trainSet.get(0) + "]");

        DataNormalization dataNormalization = new NormalizerStandardize();
        dataNormalization.fit(trainSet);
        dataNormalization.transform(trainSet);
        dataNormalization.transform(testSet);

        //System.out.println("args = [" + trainSet.get(0) + "]");
        log.info("Building Model------------------->>>>>>>>>");


        MultiLayerNetwork multiLayerNetwork = KerasModelImport
                .importKerasSequentialModelAndWeights("C:/Users/Rahul_Raj05/Downloads/Artificial_Neural_Networks/Artificial_Neural_Networks/ANN/ann_model_json",
                        "C:/Users/Rahul_Raj05/Downloads/Artificial_Neural_Networks/Artificial_Neural_Networks/ANN/ann_model");
        // multiLayerNetwork.init();
        //multiLayerNetwork.setListeners(new ScoreIterationListener());
        //DataSetIterator kFoldIterator = new KFoldIterator(trainSet);
        //multiLayerNetwork.fit(kFoldIterator,100);

        Evaluation evaluation = new Evaluation(1);
        INDArray output = multiLayerNetwork.output(testSet.getFeatureMatrix());
        output = output.cond(new AbsValueGreaterThan(0.50));
        evaluation.eval(testSet.getLabels(),output);
        System.out.println("args = [" + evaluation.stats() + "]");
    }
}