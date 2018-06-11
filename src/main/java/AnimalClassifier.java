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
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
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



        ImageRecordReader imageRecordReader = new ImageRecordReader(100,100,channels,parentPathLabelGenerator);
        imageRecordReader.initialize(trainData,null);
        DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(imageRecordReader,batchSize);
        scaler.fit(dataSetIterator);
        dataSetIterator.setPreProcessor(scaler);

    }
}
