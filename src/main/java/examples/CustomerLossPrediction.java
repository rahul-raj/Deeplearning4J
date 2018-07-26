package examples;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.util.ndarray.RecordConverter;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIteratorSplitter;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

public class CustomerLossPrediction {

    private static Logger log = LoggerFactory.getLogger("examples.CustomerLossPrediction.class");

    public TransformProcess generateSchemaAndTransform(){
        //Schema Definitions
        Schema schema = new Schema.Builder()
                .addColumnsString("RowNumber")
                .addColumnInteger("CustomerId")
                .addColumnString("Surname")
                .addColumnInteger("CreditScore")
                .addColumnCategorical("Geography",Arrays.asList("France","Spain","Germany"))
                .addColumnCategorical("Gender",Arrays.asList("Male","Female"))
                .addColumnsInteger("Age","Tenure")
                .addColumnsDouble("Balance")
                .addColumnsInteger("NumOfProducts","HasCrCard","IsActiveMember")
                .addColumnDouble("EstimatedSalary")
                .addColumnInteger("Exited")
                .build();

        //Schema Transformation
        TransformProcess transformProcess = new TransformProcess.Builder(schema)
                .removeColumns("RowNumber","Surname","CustomerId")
                .categoricalToInteger("Gender")
                .categoricalToOneHot("Geography")
                .removeColumns("Geography[France]")
                .build();
        return transformProcess;

    }

    public RecordReader generateSchemaAndReaderForPrediction(File file) throws IOException, InterruptedException {

        Schema schema = new Schema.Builder()
                .addColumnInteger("CreditScore")
                .addColumnCategorical("Geography", Arrays.asList("France","Spain","Germany"))
                .addColumnCategorical("Gender", Arrays.asList("Male","Female"))
                .addColumnsInteger("Age","Tenure")
                .addColumnDouble("Balance")
                .addColumnsInteger("NumOfProducts","HasCrCard","IsActiveMember")
                .addColumnDouble("EstimatedSalary")
                .build();
        TransformProcess transformProcess = new TransformProcess.Builder(schema)
                .categoricalToInteger("Gender")
                .categoricalToOneHot("Geography")
                .removeColumns("Geography[France]")
                .build();

        RecordReader reader = new CSVRecordReader(0,',');
        reader.initialize(new FileSplit(file));
        RecordReader recordReader = new TransformProcessRecordReader(reader,transformProcess);
        return recordReader;
    }

    public INDArray generateOutput(File file) throws IOException, InterruptedException {
        File modelFile = new File("model.zip");
        MultiLayerNetwork restored = ModelSerializer.restoreMultiLayerNetwork(modelFile);
        RecordReader recordReader = generateSchemaAndReaderForPrediction(file);
        INDArray array = RecordConverter.toArray(recordReader.next());
        NormalizerStandardize normalizerStandardize = ModelSerializer.restoreNormalizerFromFile(modelFile);
        normalizerStandardize.transform(array);
        return restored.output(array,false);
    }

    public RecordReader generateReader(File file) throws IOException, InterruptedException{
        //CSVReader - Reading from file and applying transformation
        RecordReader reader = new CSVRecordReader(1,',');
        reader.initialize(new FileSplit(file));
        RecordReader transformProcessRecordReader = new TransformProcessRecordReader(reader,generateSchemaAndTransform());
        return transformProcessRecordReader;
    }



    public static void main(String[] args) throws IOException, InterruptedException {

        CustomerLossPrediction customerLossPrediction = new CustomerLossPrediction();

        int labelIndex = 11;  // consider index 0 to 11  for input
        int numClasses = 2;
        int batchSize = 8;
        INDArray weightsArray = Nd4j.create(new double[]{0.57, 0.75});

        DataSetIterator iterator = new RecordReaderDataSetIterator(customerLossPrediction.generateReader(new ClassPathResource("Churn_Modelling.csv").getFile()),batchSize,labelIndex,numClasses);
        DataNormalization dataNormalization = new NormalizerStandardize();
        dataNormalization.fit(iterator);
        iterator.setPreProcessor(dataNormalization);
        DataSetIteratorSplitter splitter = new DataSetIteratorSplitter(iterator,1250,0.8);

        log.info("Building Model------------------->>>>>>>>>");


        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder().l2(.0005)
                .weightInit(WeightInit.RELU_UNIFORM)
                .updater(new Nesterovs(0.008,0.9)) // new Adam(0.015D); new RmsProp(0.08D)
                .list()
                .layer(new DenseLayer.Builder().nIn(11).nOut(8).activation(Activation.RELU).dropOut(0.9).build())
                .layer(new DenseLayer.Builder().nIn(8).nOut(6).activation(Activation.RELU).dropOut(0.9).build())
                .layer(new DenseLayer.Builder().nIn(6).nOut(6).activation(Activation.RELU).dropOut(0.9).build())
                .layer(new OutputLayer.Builder(new LossMCXENT(weightsArray))
                        .nIn(6).nOut(2).activation(Activation.SOFTMAX).build())
                .backprop(true).pretrain(false)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();

        model.setListeners(new ScoreIterationListener(100));

        //Benchmarking: Reduce/Disable Frequent calls to garbage collector
        //Nd4j.getMemoryManager().setAutoGcWindow(10000);
        //Nd4j.getMemoryManager().togglePeriodicGc(false);

        model.fit(splitter.getTrainIterator(),100);
        Evaluation evaluation = model.evaluate(splitter.getTestIterator(),Arrays.asList("0","1"));
        System.out.println("args = " + evaluation.stats() + "");

        File file = new File("model.zip");
        ModelSerializer.writeModel(model,file,true);
        ModelSerializer.addNormalizerToModel(file,dataNormalization);
        MultiLayerNetwork restored = ModelSerializer.restoreMultiLayerNetwork(new File("model.zip"));


        System.out.println(restored.params()+" \n"+restored.getLayerWiseConfigurations());
        INDArray output = customerLossPrediction.generateOutput(new File("test.csv"));
        //INDArray array = restored.output(output,false);
        log.info(output.toString());



    }
}