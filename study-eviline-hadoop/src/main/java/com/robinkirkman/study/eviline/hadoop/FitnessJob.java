package com.robinkirkman.study.eviline.hadoop;

import java.io.IOException;

import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;

public class FitnessJob {
  public static final String GARBAGE_HEIGHT = FitnessJob.class.getName() + ".garbage_height";
  public static final String LOOKAHEAD = FitnessJob.class.getName() + ".lookahead";
  public static final String RETRIES = FitnessJob.class.getName() + ".retries";
  public static final String GENERATION_SIZE = FitnessJob.class.getName() + ".generation_size";
  public static final String MUTATIONS = FitnessJob.class.getName() + ".mutations";

  public static final int GARBAGE_HEIGHT_DEFAULT = 10;
  public static final int LOOKAHEAD_DEFAULT = 3;
  public static final int RETRIES_DEFAULT = 100;
  public static final int GENERATION_SIZE_DEFAULT = 100;
  public static final int MUTATIONS_DEFAULT = 100;

  public static void setGarbageHeight(Job job, int garbageHeight) {
    job.getConfiguration().setInt(GARBAGE_HEIGHT, garbageHeight);
  }

  public static void setLookahead(Job job, int lookahead) {
    job.getConfiguration().setInt(LOOKAHEAD, lookahead);
  }

  public static void setRetries(Job job, int retries) {
    job.getConfiguration().setInt(RETRIES, retries);
  }

  public static void setGenerationSize(Job job, int generationSize) {
    job.getConfiguration().setInt(GENERATION_SIZE, generationSize);
  }

  public static void setMutations(Job job, int mutations) {
    job.getConfiguration().setInt(MUTATIONS, mutations);
  }

  public static void configureJob(Job job) {
    TextInputFormat.setMinInputSplitSize(job, 1);
    TextInputFormat.setMaxInputSplitSize(job, 128);
    
    job.setInputFormatClass(FitnessFormat.FitnessInputFormat.class);
    job.setMapOutputKeyClass(NullWritable.class);
    job.setMapOutputValueClass(FitnessCoefficientsResult.class);
    job.setMapperClass(FitnessMapper.class);
    job.setNumReduceTasks(1);
    job.setOutputKeyClass(NullWritable.class);
    job.setOutputValueClass(FitnessCoefficientsResult.class);
    job.setReducerClass(FitnessReducer.class);
    job.setOutputFormatClass(FitnessFormat.FitnessOutputFormat.class);
  }

  public static class FitnessMapper
      extends Mapper<FitnessCoefficients, NullWritable, NullWritable, FitnessCoefficientsResult> {
    private FitnessTask.FitnessMapper mapper;

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
      int garbageHeight = context.getConfiguration().getInt(GARBAGE_HEIGHT, GARBAGE_HEIGHT_DEFAULT);
      int lookahead = context.getConfiguration().getInt(LOOKAHEAD, LOOKAHEAD_DEFAULT);
      int retries = context.getConfiguration().getInt(RETRIES, RETRIES_DEFAULT);
      int generationSize = context.getConfiguration().getInt(GENERATION_SIZE, GENERATION_SIZE_DEFAULT);

      mapper = new FitnessTask.FitnessMapper(garbageHeight, lookahead, retries, generationSize);
      mapper.setup();
    }

    @Override
    protected void map(FitnessCoefficients key, NullWritable value, Context context)
        throws IOException, InterruptedException {
      mapper.map(key);
    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
      mapper.cleanup((v) -> context.write(NullWritable.get(), v));
    }
  }

  public static class FitnessReducer extends Reducer<NullWritable, FitnessCoefficientsResult, NullWritable, FitnessCoefficientsResult> {
    private FitnessTask.FitnessReducer reducer;

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
      int generationSize = context.getConfiguration().getInt(GENERATION_SIZE, GENERATION_SIZE_DEFAULT);
      reducer = new FitnessTask.FitnessReducer(generationSize);
      reducer.setup();
    }

    @Override
    protected void reduce(NullWritable key, Iterable<FitnessCoefficientsResult> values, Context context)
        throws IOException, InterruptedException {
      for (FitnessCoefficientsResult value : values) {
        reducer.reduce(value);
      }
    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
      reducer.cleanup((v) -> context.write(NullWritable.get(), v));
    }
  }
}
