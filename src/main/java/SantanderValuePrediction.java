import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.analysis.DataAnalysis;
import org.datavec.api.transform.analysis.columns.ColumnAnalysis;
import org.datavec.api.transform.analysis.columns.LongAnalysis;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.AnalyzeSpark;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.List;

public class SantanderValuePrediction {

    private static Logger log = LoggerFactory.getLogger("SantanderValuePrediction.class");

    public static void main(String[] args) throws IOException, InterruptedException {


        Schema schema = new Schema.Builder()
                                  .addColumnsString("ID")
                                  .addColumnLong("target")
                                  .addColumnsLong("col_%d",0,4990)
                                  .build();
        TransformProcess transformProcess = new TransformProcess.Builder(schema)
                                                                .removeColumns("ID")
                                                                .build();
        RecordReader recordReader = new CSVRecordReader(1,',');
        recordReader.initialize(new FileSplit(new ClassPathResource("train.csv").getFile()));
        TransformProcessRecordReader transformProcessRecordReader = new TransformProcessRecordReader(recordReader,transformProcess);


        SparkConf sparkConf = new SparkConf();
        sparkConf.setMaster("local[*]");
        sparkConf.setAppName("Santander App");
        JavaSparkContext javaSparkContext = new JavaSparkContext(sparkConf);
        JavaRDD<String> directory = javaSparkContext.textFile(new ClassPathResource("train.csv").getFile().getParent());
        JavaRDD<List<Writable>> parsedData = directory.map(new StringToWritablesFunction(transformProcessRecordReader));

        DataAnalysis dataAnalysis = AnalyzeSpark.analyze(schema,parsedData);
        System.out.println("args = [" + dataAnalysis + "]");







/*        TransformProcessRecordReader transformProcessRecordReader = new TransformProcessRecordReader(recordReader,transformProcess);
        //System.out.println("args = [" + recordReader.next().get(1).toString() + "]");
        DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(transformProcessRecordReader,10,0,0,true);
        System.out.println("args = [" + dataSetIterator.next()+ "]");
        */


     //   RecordReader transformProcessRecordReader = new TransformProcessRecordReader(recordReader,transformProcess);

      //  int batchSize=10;
      //  DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(recordReader,10,1,2,true);
      //  System.out.println("args = [" + dataSetIterator.next() + "]");

    }
}
