import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIteratorSplitter;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Arrays;

public class DeepLearning4j {

    private static Logger log = LoggerFactory.getLogger("DeepLearning4j.class");

    public static void main(String[] args) throws IOException, InterruptedException {

        //Schema Definitions
        Schema schema = new Schema.Builder()
                .addColumnsString("RowNumber")
                .addColumnInteger("CustomerId")
                .addColumnString("Surname")
                .addColumnInteger("CreditScore")
                .addColumnCategorical("Geography",Arrays.asList("France","Spain","Germany"))
                .addColumnCategorical("Gender",Arrays.asList("Male","Female"))
                .addColumnsInteger("Age","Tenure","Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary","Exited").build();

        //Schema Transformation
        TransformProcess transformProcess = new TransformProcess.Builder(schema)
                .removeColumns("RowNumber","Surname","CustomerId")
                .categoricalToInteger("Gender")
                .categoricalToOneHot("Geography")
                .removeColumns("Geography[France]")
                .build();

        //CSVReader - Reading from file and applying transformation
        RecordReader reader = new CSVRecordReader(1,',');
        reader.initialize(new FileSplit(new ClassPathResource("Churn_Modelling.csv").getFile()));
        RecordReader transformProcessRecordReader = new TransformProcessRecordReader(reader,transformProcess);

        int labelIndex = 11;  // consider index 0 to 11  for input
        int numClasses = 1;
        int batchSize = 100;

        DataSetIterator iterator = new RecordReaderDataSetIterator(transformProcessRecordReader,batchSize,labelIndex,numClasses);
        DataSetIteratorSplitter splitter = new DataSetIteratorSplitter(iterator,100,0.8);

        log.info("Building Model------------------->>>>>>>>>");

        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new Adam(0.01))
                .list()
                .layer(new DenseLayer.Builder().nIn(11).nOut(6).weightInit(WeightInit.DISTRIBUTION).dist(new UniformDistribution(-0.05,0.05)).activation(Activation.RELU).dropOut(0.9).build())
                .layer(new DenseLayer.Builder().nIn(6).nOut(6).weightInit(WeightInit.DISTRIBUTION).dist(new UniformDistribution(-0.05,0.05)).activation(Activation.RELU).dropOut(0.9).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .nIn(6).nOut(1).weightInit(WeightInit.UNIFORM).activation(Activation.SIGMOID).build())
                .backprop(true).pretrain(false)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();
        model.setListeners(new ScoreIterationListener(100));
        model.fit(splitter.getTrainIterator(),100);
        Evaluation evaluation = model.evaluate(splitter.getTestIterator(),Arrays.asList("0","1"));
        System.out.println("args = " + evaluation.stats() + "");


      //  Evaluation evaluation = new Evaluation(1);
       // INDArray output = model.output(splitter.getTestIterator());


/*
        Evaluation evaluation = new Evaluation(1);
        INDArray output = model.output(splitter.getTestIterator());
       // output = output.cond(new AbsValueGreaterThan(0.50));
        DataSetIterator splitIterator = splitter.getTestIterator();
        List<DataSet> dataset = new ArrayList<>();
        while(iterator.hasNext()){
             dataset.add(splitIterator.next());
        }
        DataSet testSet = DataSet.merge(dataset);
        evaluation.eval(testSet.getLabels(),output);
        System.out.println("args = " + evaluation.stats() + "");
*/


    }
}