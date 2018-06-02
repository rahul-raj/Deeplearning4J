import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.deeplearning4j.arbiter.MultiLayerSpace;
import org.deeplearning4j.arbiter.conf.updater.AdamSpace;
import org.deeplearning4j.arbiter.layers.DenseLayerSpace;
import org.deeplearning4j.arbiter.layers.OutputLayerSpace;
import org.deeplearning4j.arbiter.optimize.api.CandidateGenerator;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.data.DataSetIteratorFactoryProvider;
import org.deeplearning4j.arbiter.optimize.generator.RandomSearchGenerator;
import org.deeplearning4j.arbiter.optimize.parameter.continuous.ContinuousParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.integer.IntegerParameterSpace;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIteratorSplitter;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Arrays;
import java.util.Map;

public class HyperParamTuning {

    private static Logger log = LoggerFactory.getLogger("DeepLearning4j.class");

    public static void main(String[] args) throws IOException, InterruptedException {



        ParameterSpace<Double> learningRateParam = new ContinuousParameterSpace(0.0001,0.01);
        ParameterSpace<Integer> layerSizeParam = new IntegerParameterSpace(15,300);

        MultiLayerSpace hyperParamaterSpace = new MultiLayerSpace.Builder()
                                                  .updater(new AdamSpace(learningRateParam))
                                                  .addLayer(new DenseLayerSpace.Builder()
                                                          .activation(Activation.RELU)
                                                          .nIn(layerSizeParam)
                                                          .nOut(layerSizeParam)
                                                          .build())
                                                  .addLayer(new DenseLayerSpace.Builder()
                                                          .activation(Activation.RELU)
                                                          .nIn(layerSizeParam)
                                                          .nOut(layerSizeParam)
                                                          .build())
                                                  .addLayer(new OutputLayerSpace.Builder()
                                                          .activation(Activation.SIGMOID)
                                                          .lossFunction(LossFunctions.LossFunction.XENT)
                                                          .nOut(1)
                                                          .build())
                                                  .build();
        CandidateGenerator candidateGenerator = new RandomSearchGenerator(hyperParamaterSpace);
        DataProvider dataProvider = new ExampleDataProvider();











    }


      private static class ExampleDataProvider implements DataProvider{

         public RecordReader dataPreprocess() throws IOException, InterruptedException {
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
             return transformProcessRecordReader;
         }

          @Override
          public DataSetIterator trainData(Map<String, Object> dataParameters) {
             try{
                 if(dataParameters!=null && !dataParameters.isEmpty()){
                     int labelIndex = 11;  // consider index 0 to 11  for input
                     int numClasses = 1;
                     int batchSize = (Integer) dataParameters.get("batchSize");
                     DataSetIterator iterator = new RecordReaderDataSetIterator(dataPreprocess(),batchSize,labelIndex,numClasses);
                     DataSetIteratorSplitter splitter = new DataSetIteratorSplitter(iterator,1000,0.8);
                     return splitter.getTrainIterator();
                 }
                 return null;
             }
             catch(Exception e){
                 throw new RuntimeException();
             }

          }

          @Override
          public DataSetIterator testData(Map<String, Object> dataParameters) {
              try{
                  if(dataParameters!=null && !dataParameters.isEmpty()){
                      int labelIndex = 11;  // consider index 0 to 11  for input
                      int numClasses = 1;
                      int batchSize = (Integer) dataParameters.get("batchSize");
                      DataSetIterator iterator = new RecordReaderDataSetIterator(dataPreprocess(),batchSize,labelIndex,numClasses);
                      DataSetIteratorSplitter splitter = new DataSetIteratorSplitter(iterator,1000,0.8);
                      return splitter.getTestIterator();
                  }
                  return null;
              }
              catch(Exception e){
                  throw new RuntimeException();
              }
          }

          @Override
          public Class<?> getDataType() {
              return RecordReaderDataSetIterator.class;
          }
      }
}