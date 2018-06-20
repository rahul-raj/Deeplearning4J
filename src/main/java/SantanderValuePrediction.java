import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

public class SantanderValuePrediction {

    private static Logger log = LoggerFactory.getLogger("SantanderValuePrediction.class");

    public static void main(String[] args) throws IOException, InterruptedException {



        RecordReader recordReader = new CSVRecordReader(1,',');
        recordReader.initialize(new FileSplit(new File("C:/Users/Admin/Downloads/test/train_modified.csv")));
        //System.out.println("args = [" + recordReader.next().get(1).toString() + "]");
        DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(recordReader,10,0,0,true);
        System.out.println("args = [" + dataSetIterator.next()+ "]");


     //   RecordReader transformProcessRecordReader = new TransformProcessRecordReader(recordReader,transformProcess);

      //  int batchSize=10;
      //  DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(recordReader,10,1,2,true);
      //  System.out.println("args = [" + dataSetIterator.next() + "]");

    }
}
