package examples;

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
import org.deeplearning4j.arbiter.optimize.api.saving.ResultSaver;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxCandidatesCondition;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxTimeCondition;
import org.deeplearning4j.arbiter.optimize.api.termination.TerminationCondition;
import org.deeplearning4j.arbiter.optimize.config.OptimizationConfiguration;
import org.deeplearning4j.arbiter.optimize.generator.RandomSearchGenerator;
import org.deeplearning4j.arbiter.optimize.parameter.continuous.ContinuousParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.integer.IntegerParameterSpace;
import org.deeplearning4j.arbiter.optimize.runner.IOptimizationRunner;
import org.deeplearning4j.arbiter.optimize.runner.LocalOptimizationRunner;
import org.deeplearning4j.arbiter.optimize.runner.listener.impl.LoggingStatusListener;
import org.deeplearning4j.arbiter.saver.local.FileModelSaver;
import org.deeplearning4j.arbiter.scoring.impl.EvaluationScoreFunction;
import org.deeplearning4j.arbiter.task.MultiLayerNetworkTaskCreator;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;

public class HyperParamTuning {

    private static Logger log = LoggerFactory.getLogger("examples.HyperParamTuning.class");

    public static void main(String[] args) throws IOException, InterruptedException {



        ParameterSpace<Double> learningRateParam = new ContinuousParameterSpace(0.0001,0.01);
        ParameterSpace<Integer> layerSizeParam = new IntegerParameterSpace(5,11);
        MultiLayerSpace hyperParamaterSpace = new MultiLayerSpace.Builder()
                                                  .updater(new AdamSpace(learningRateParam))
                                                //  .weightInit(WeightInit.DISTRIBUTION).dist(new LogNormalDistribution())
                                                  .addLayer(new DenseLayerSpace.Builder()
                                                          .activation(Activation.RELU)
                                                          .nIn(11)
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

        Map<String,Object> dataParams = new HashMap<>();
        dataParams.put("batchSize",new Integer(10));

        Map<String,Object> commands = new HashMap<>();
        commands.put(DataSetIteratorFactoryProvider.FACTORY_KEY,ExampleDataProvider.class.getCanonicalName());

        CandidateGenerator candidateGenerator = new RandomSearchGenerator(hyperParamaterSpace,dataParams);
        DataProvider dataProvider = new ExampleDataProvider(dataParams);
        ResultSaver modelSaver = new FileModelSaver("resources/");
        ScoreFunction scoreFunction = new EvaluationScoreFunction(Evaluation.Metric.ACCURACY);


        TerminationCondition[] conditions = {
                new MaxTimeCondition(120, TimeUnit.MINUTES),
                new MaxCandidatesCondition(30)

        };

        OptimizationConfiguration optimizationConfiguration = new OptimizationConfiguration.Builder()
                                                               .candidateGenerator(candidateGenerator)
                                                               .dataProvider(dataProvider)
                                                               .modelSaver(modelSaver)
                                                               .scoreFunction(scoreFunction)
                                                               .terminationConditions(conditions)
                                                               .build();

        IOptimizationRunner runner = new LocalOptimizationRunner(optimizationConfiguration,new MultiLayerNetworkTaskCreator());
        //Uncomment this if you want to store the model.
       // StatsStorage ss = new FileStatsStorage(new File("HyperParamOptimizationStats.dl4j"));
        runner.addListeners(new LoggingStatusListener()); //new ArbiterStatusListener(ss)
        runner.execute();

        //Print the best hyper params

        double bestScore = runner.bestScore();
        int bestCandidateIndex = runner.bestScoreCandidateIndex();
        int numberOfConfigsEvaluated = runner.numCandidatesCompleted();

        String s = "Best score: " + bestScore + "\n" +
                "Index of model with best score: " + bestCandidateIndex + "\n" +
                "Number of configurations evaluated: " + numberOfConfigsEvaluated + "\n";

        System.out.println(s);

    }


      private static class ExampleDataProvider implements DataProvider{

         final int labelIndex = 11;  // consider index 0 to 11  for input
         final int numClasses = 1;

         private Map<String,Object> dataParams;

         public ExampleDataProvider(Map<String,Object> dataParams){
            this.dataParams = dataParams;
         }

         public DataSetIteratorSplitter dataSplit(DataSetIterator iterator) throws IOException, InterruptedException {
             DataNormalization dataNormalization = new NormalizerStandardize();
             dataNormalization.fit(iterator);
             iterator.setPreProcessor(dataNormalization);
             DataSetIteratorSplitter splitter = new DataSetIteratorSplitter(iterator,1000,0.8);
             return splitter;
         }

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
          public DataSetIterator trainData(Map<String, Object> dataParameters){
             try{
                 if(dataParameters!=null && !dataParameters.isEmpty()){
                     if(dataParameters.containsKey("batchSize")){
                         int batchSize = (Integer) dataParameters.get("batchSize");
                         DataSetIterator iterator = new RecordReaderDataSetIterator(dataPreprocess(),batchSize,labelIndex,numClasses);
                         return dataSplit(iterator).getTestIterator();
                     }
                 }
                 else
                   return null;
             }
             catch(Exception e){
                 throw new RuntimeException();
             }
             return null;
          }

          @Override
          public DataSetIterator testData(Map<String, Object> dataParameters) {
              try{
                  if(dataParameters!=null && !dataParameters.isEmpty()){
                      if(dataParameters.containsKey("batchSize")){
                          int batchSize = (Integer) dataParameters.get("batchSize");
                          DataSetIterator iterator = new RecordReaderDataSetIterator(dataPreprocess(),batchSize,labelIndex,numClasses);
                          return dataSplit(iterator).getTestIterator();
                      }
                  }
                  return null;
              }
              catch(Exception e){
                  throw new RuntimeException();
              }
          }

          @Override
          public Class<?> getDataType() {
              return DataSetIterator.class;
          }

          @Override
          public String toString() {
              return "ExampleDataProvider()";
          }
      }
}