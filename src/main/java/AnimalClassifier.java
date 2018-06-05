import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class AnimalClassifier {
    public static void main(String[] args) throws Exception {

        int channels = 3;
        int batchSize=10;

        File parentDir = new File("C:/Users/Admin/Downloads/imagenet");
        FileSplit fileSplit = new FileSplit(parentDir, NativeImageLoader.ALLOWED_FORMATS,new Random(42));
        ParentPathLabelGenerator parentPathLabelGenerator = new ParentPathLabelGenerator();

        BalancedPathFilter balancedPathFilter = new BalancedPathFilter(new Random(42),NativeImageLoader.ALLOWED_FORMATS,parentPathLabelGenerator);
        InputSplit[] inputSplits = fileSplit.sample(balancedPathFilter,80,20);
        //System.out.println("args = [" + inputSplits[0].length() + "]"+inputSplits[1].length()+" ");
        InputSplit trainData = inputSplits[0];
        InputSplit testData = inputSplits[1];

        ImageTransform transform1 = new FlipImageTransform(new Random(42));
        ImageTransform transform2 = new FlipImageTransform(new Random(123));
        ImageTransform transform3 = new WarpImageTransform(new Random(42),42);

        List<ImageTransform> transformList = new ArrayList<>();
        transformList.add(transform1);
        transformList.add(transform2);
        transformList.add(transform3);

        DataNormalization scaler = new ImagePreProcessingScaler(0,1);


        ImageRecordReader imageRecordReader = new ImageRecordReader(100,100,channels,parentPathLabelGenerator);
        imageRecordReader.initialize(trainData,null);
        DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(imageRecordReader,batchSize);

    }
}
