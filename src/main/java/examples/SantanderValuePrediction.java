package examples;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.List;

public class SantanderValuePrediction {

    private static Logger log = LoggerFactory.getLogger("examples.SantanderValuePrediction.class");


    public static void main(String[] args) throws IOException, InterruptedException, IllegalAccessException, InstantiationException {

        Schema schema = new Schema.Builder()
                                  .addColumnsString("ID")
                                  .addColumnLong("target")
                                  .addColumnsDouble("col_%d",0,4990)
                                  .build();
        TransformProcess transformProcess = new TransformProcess.Builder(schema)
                                                                .removeColumns("ID")
                                                                .build();
        RecordReader recordReader = new CSVRecordReader(1,',');
        recordReader.initialize(new FileSplit(new ClassPathResource("train.csv").getFile()));
        TransformProcessRecordReader transformProcessRecordReader = new TransformProcessRecordReader(recordReader,transformProcess);

        log.info("Data Pre-processing starts...");

        double[] max = new double[4992];
        double[] min = new double[4992];
        while(transformProcessRecordReader.hasNext()){
               List<Writable> record = transformProcessRecordReader.next();
               for(int i=1;i<=4991;i++){
                     max[i]=Math.max(max[i],record.get(i).toDouble());
                     min[i]=Math.min(min[i],record.get(i).toDouble());
               }
        }

        recordReader=new CSVRecordReader(1,',');
        recordReader.initialize(new FileSplit(new ClassPathResource("train.csv").getFile()));
        TransformProcess.Builder builder = new TransformProcess.Builder(schema)
                                                               .removeColumns("ID");
        log.info("Removing constant features...");
        for(int i=1;i<=4991;i++){
            if(max[i]==min[i]){
                builder.removeColumns("col_"+String.valueOf(i));
            }
        }

 
        transformProcessRecordReader = new TransformProcessRecordReader(recordReader,builder.build());
        System.out.println("args = [" + transformProcessRecordReader.next().size() + "]");






    }
}
