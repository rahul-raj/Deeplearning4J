import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;

import java.io.File;
import java.util.Random;

public class AnimalClassifier {
    public static void main(String[] args) {

        File parentDir = new File("C:/Users/Admin/Downloads/imagenet");
        FileSplit fileSplit = new FileSplit(parentDir, NativeImageLoader.ALLOWED_FORMATS,new Random(42));
        ParentPathLabelGenerator parentPathLabelGenerator = new ParentPathLabelGenerator();

        BalancedPathFilter balancedPathFilter = new BalancedPathFilter(new Random(42),NativeImageLoader.ALLOWED_FORMATS,parentPathLabelGenerator);
        InputSplit[] inputSplits = fileSplit.sample(balancedPathFilter,80,20);
        //System.out.println("args = [" + inputSplits[0].length() + "]"+inputSplits[1].length()+" ");
        InputSplit trainData = inputSplits[0];
        InputSplit testData = inputSplits[1];
    }
}
