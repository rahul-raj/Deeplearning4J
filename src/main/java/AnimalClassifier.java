import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.PipelineImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class AnimalClassifier {
    public static void main(String[] args) throws Exception {

        //R,G,B channels
        int channels = 3;
        int batchSize=10;

        //load files and split
        File parentDir = new File("C:/Users/Admin/Downloads/imagenet");
        FileSplit fileSplit = new FileSplit(parentDir, NativeImageLoader.ALLOWED_FORMATS,new Random(42));
        int numLabels = fileSplit.getRootDir().listFiles(File::isDirectory).length;

        //identify labels in the path
        ParentPathLabelGenerator parentPathLabelGenerator = new ParentPathLabelGenerator();

        //file split to train/test using the weights.
        BalancedPathFilter balancedPathFilter = new BalancedPathFilter(new Random(42),NativeImageLoader.ALLOWED_FORMATS,parentPathLabelGenerator);
        InputSplit[] inputSplits = fileSplit.sample(balancedPathFilter,80,20);

        //get train/test data
        InputSplit trainData = inputSplits[0];
        InputSplit testData = inputSplits[1];

        //Data augmentation
        ImageTransform transform1 = new FlipImageTransform(new Random(42));
        ImageTransform transform2 = new FlipImageTransform(new Random(123));
        ImageTransform transform3 = new WarpImageTransform(new Random(42),42);

        //pipelines to specify image transformation. 
        List<Pair<ImageTransform,Double>> pipeline = Arrays.asList(
                new Pair<>(transform1, 0.8),
                new Pair<>(transform2, 0.7),
                new Pair<>(transform3, 0.5)
        );

        ImageTransform transform = new PipelineImageTransform(pipeline);
        DataNormalization scaler = new ImagePreProcessingScaler(0,1);

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                                             .weightInit(WeightInit.XAVIER)
                                             .updater(new Nesterovs(0.008D,0.9D))
                                             .list()
                                             .layer(new ConvolutionLayer.Builder(5,5)
                                                        .nIn(channels)
                                                        .nOut(30)
                                                        .stride(1,1)
                                                        .activation(Activation.RELU)
                                                        .build())
                                             .layer(new SubsamplingLayer.Builder(PoolingType.MAX)
                                                        .stride(2,2)
                                                        .kernelSize(2,2)
                                                        .build())
                                             .layer(new ConvolutionLayer.Builder(5,5)
                                                        .nOut(30)
                                                        .stride(1,1)
                                                        .activation(Activation.RELU)
                                                        .build())
                                             .layer(new SubsamplingLayer.Builder(PoolingType.MAX)
                                                        .stride(2,2)
                                                        .kernelSize(2,2)
                                                        .build())
                                             .layer(new DenseLayer.Builder()
                                                        .nOut(500)
                                                        .activation(Activation.RELU)
                                                        .build())
                                             .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                                        .nOut(numLabels)
                                                        .activation(Activation.SOFTMAX)
                                                        .build())
                                             .setInputType(InputType.convolutionalFlat(30,30,3))
                                             .backprop(true).pretrain(false)
                                             .build();


        //train without transformations
        ImageRecordReader imageRecordReader = new ImageRecordReader(30,30,channels,parentPathLabelGenerator);
        imageRecordReader.initialize(trainData,null);
        DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(imageRecordReader,batchSize,1,numLabels);
        scaler.fit(dataSetIterator);
        dataSetIterator.setPreProcessor(scaler);
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        model.setListeners(new PerformanceListener(100)); //PerformanceListener for optimized training
        model.fit(dataSetIterator,100);

        //train with transformations
        imageRecordReader.initialize(trainData,transform);
        dataSetIterator = new RecordReaderDataSetIterator(imageRecordReader,batchSize,1,numLabels);
        scaler.fit(dataSetIterator);
        dataSetIterator.setPreProcessor(scaler);
        model.fit(dataSetIterator,100);

        imageRecordReader.initialize(testData);
        dataSetIterator = new RecordReaderDataSetIterator(imageRecordReader,batchSize,1,numLabels);
        scaler.fit(dataSetIterator);
        dataSetIterator.setPreProcessor(scaler);

        Evaluation evaluation = model.evaluate(dataSetIterator);
        System.out.println("args = [" + evaluation.stats() + "]");

        ModelSerializer.writeModel(model,new File("cnntrainedmodel.zip"),true);



    }
}
