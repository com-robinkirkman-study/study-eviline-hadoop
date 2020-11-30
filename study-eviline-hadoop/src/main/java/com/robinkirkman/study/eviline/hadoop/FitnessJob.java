package com.robinkirkman.study.eviline.hadoop;

import java.io.ByteArrayOutputStream;
import java.io.DataOutput;
import java.io.DataOutputStream;
import java.io.IOException;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.TreeMap;

import org.apache.avro.hadoop.io.AvroDatumConverterFactory.AvroWrapperConverter;
import org.apache.avro.mapred.AvroKey;
import org.apache.avro.mapred.AvroValue;
import org.apache.curator.shaded.com.google.common.primitives.Bytes;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.RawComparator;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputFormat;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.LineRecordReader;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.eviline.core.Block;
import org.eviline.core.Engine;
import org.eviline.core.Field;
import org.eviline.core.ai.AIPlayer;
import org.eviline.core.ai.DefaultAIKernel;
import org.eviline.core.ai.DefaultFitness;
import org.eviline.core.ai.Player;

public class FitnessJob {
  public static final String GARBAGE_HEIGHT = FitnessJob.class.getName() + ".garbage_height";
  public static final String LOOKAHEAD = FitnessJob.class.getName() + ".lookahead";
  public static final String RETRIES = FitnessJob.class.getName() + ".retries";
  public static final String GENERATION_SIZE = FitnessJob.class.getName() + ".generation_size";

  public static void configureJob(Job job) {
    job.setInputFormatClass(FitnessInputFormat.class);
    job.setMapOutputKeyClass(NullWritable.class);
    job.setMapOutputValueClass(FitnessResultCoefficients.class);
    job.setMapperClass(FitnessMapper.class);
    job.setNumReduceTasks(1);
    job.setOutputKeyClass(NullWritable.class);
    job.setOutputValueClass(Text.class);
    job.setReducerClass(FitnessReducer.class);
    job.setOutputFormatClass(TextOutputFormat.class);
  }

  /**
   * Comparator that places the worst results first, preceded by nulls.
   * 
   * @author robin
   *
   */
  public static class FitnessComparator implements Comparator<FitnessResultCoefficients> {
    @Override
    public int compare(FitnessResultCoefficients lhs, FitnessResultCoefficients rhs) {
      if (lhs.getResult() == null && rhs.getResult() == null)
        return 0;
      else if (lhs.getResult() == null)
        return -1;
      else if (rhs.getResult() == null)
        return 1;

      int c = -Long.compare(lhs.getResult().getRemainingGarbage(), rhs.getResult().getRemainingGarbage());
      if (c != 0)
        return c;
      return -Long.compare(lhs.getResult().getLinesCleared(), rhs.getResult().getLinesCleared());
    }
  }

  private static class FitnessRecordReader extends RecordReader<FitnessCoefficients, NullWritable> {
    private RecordReader<LongWritable, Text> lines;
    private FitnessCoefficients key;

    public FitnessRecordReader(RecordReader<LongWritable, Text> lines) {
      this.lines = lines;
      key = new FitnessCoefficients(new ArrayList<>());
    }

    @Override
    public void initialize(InputSplit split, TaskAttemptContext context) throws IOException, InterruptedException {
      lines.initialize(split, context);
    }

    @Override
    public boolean nextKeyValue() throws IOException, InterruptedException {
      while (lines.nextKeyValue()) {
        String line = lines.getCurrentValue().toString();
        if (line.contains("#"))
          line = line.substring(0, line.indexOf('#'));
        String[] coefficients = line.split(",");
        if (coefficients.length == 1 && coefficients[0].trim().isEmpty())
          continue;
        key.getCoefficients().clear();
        for (String c : coefficients) {
          key.getCoefficients().add(Double.parseDouble(c));
        }
        return true;
      }
      return false;
    }

    @Override
    public FitnessCoefficients getCurrentKey() throws IOException, InterruptedException {
      return key;
    }

    @Override
    public NullWritable getCurrentValue() throws IOException, InterruptedException {
      return NullWritable.get();
    }

    @Override
    public float getProgress() throws IOException, InterruptedException {
      return lines.getProgress();
    }

    @Override
    public void close() throws IOException {
      lines.close();
    }
  }

  public static class FitnessInputFormat extends InputFormat<FitnessCoefficients, NullWritable> {
    private TextInputFormat text = new TextInputFormat();

    @Override
    public List<InputSplit> getSplits(JobContext context) throws IOException, InterruptedException {
      return text.getSplits(context);
    }

    @Override
    public RecordReader<FitnessCoefficients, NullWritable> createRecordReader(InputSplit split,
        TaskAttemptContext context) throws IOException, InterruptedException {
      return new FitnessRecordReader(text.createRecordReader(split, context));
    }
  }

  public static class FitnessMapper
      extends Mapper<FitnessCoefficients, NullWritable, NullWritable, FitnessResultCoefficients> {
    private static Block GARBAGE_BLOCK = new Block(Block.MASK_GARBAGE);
    
    private static long countGarbageLines(Field field) {
      long garbageLines = 0;
      for (int y = 0; y < field.HEIGHT; ++y) {
        for (int x = 0; x < field.WIDTH; ++x) {
          if (field.block(x, y) == GARBAGE_BLOCK) {
            ++garbageLines;
            break;
          }
        }
      }
      return garbageLines;
    }

    private int garbageHeight;
    private int lookahead;
    private int retries;
    private int generationSize;

    private DefaultFitness fitness = new DefaultFitness();
    private DefaultAIKernel aiKernel = new DefaultAIKernel(fitness);

    private PriorityQueue<FitnessResultCoefficients> best;

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
      garbageHeight = context.getConfiguration().getInt(GARBAGE_HEIGHT, -1);
      lookahead = context.getConfiguration().getInt(LOOKAHEAD, -1);
      retries = context.getConfiguration().getInt(RETRIES, -1);
      generationSize = context.getConfiguration().getInt(GENERATION_SIZE, -1);
      best = new PriorityQueue<>(generationSize, new FitnessComparator());
    }

    @Override
    protected void map(FitnessCoefficients key, NullWritable value, Context context)
        throws IOException, InterruptedException {
      Random random = new SecureRandom(Bytes.toArray(key.getCoefficients()));

      FitnessResultCoefficients out = new FitnessResultCoefficients();
      out.setResult(new FitnessResult());
      out.setCoefficients(new FitnessCoefficients(new ArrayList<>(key.getCoefficients())));

      long linesCleared = 0, remainingGarbage = 0;

      for (int i = 0; i < retries; ++i) {
        Field field = new Field();
        for (int dy = 0; dy < garbageHeight; ++dy) {
          int y = (Field.BUFFER + field.HEIGHT - 1) - dy;
          for (int x = 0; x < field.WIDTH; ++x) {
            if (random.nextBoolean()) {
              field.setBlock(x, y, GARBAGE_BLOCK);
            }
          }
        }

        Engine engine = new Engine(field,
            new org.eviline.core.Configuration(/* downFrames= */1, /* respawnFrames= */1));
        Player player = new AIPlayer(aiKernel, engine, lookahead);

        while (!engine.isOver()) {
          engine.tick(player.tick());
          if (countGarbageLines(field) == 0)
            break;
        }

        linesCleared += engine.getLines();
        remainingGarbage += countGarbageLines(field);
      }

      out.getResult().setLinesCleared(linesCleared);
      out.getResult().setRemainingGarbage(remainingGarbage);

      best.offer(out);
      if (best.size() > generationSize)
        best.poll();
    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
      while (!best.isEmpty()) {
        context.write(NullWritable.get(), best.poll());
      }
    }
  }

  public static class FitnessReducer extends Reducer<NullWritable, FitnessResultCoefficients, NullWritable, Text> {
    private int generationSize;
    private PriorityQueue<FitnessResultCoefficients> best;

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
      generationSize = context.getConfiguration().getInt(GENERATION_SIZE, -1);
    }

    @Override
    protected void reduce(NullWritable key, Iterable<FitnessResultCoefficients> values, Context context)
        throws IOException, InterruptedException {
      for (FitnessResultCoefficients out : values) {
        best.offer(out);
        if (best.size() > generationSize)
          best.poll();
      }
    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
      ArrayList<String> lines = new ArrayList<>(best.size());
      while (!best.isEmpty()) {
        FitnessResultCoefficients out = best.poll();
        StringBuilder line = new StringBuilder();
        for (Double d : out.getCoefficients().getCoefficients()) {
          if (line.length() > 0)
            line.append(", ");
          line.append(d);
        }
        line.append(" # remaining_garbage: ");
        line.append(out.getResult().getRemainingGarbage());
        line.append(" lines_cleared: ");
        line.append(out.getResult().getLinesCleared());
        lines.add(line.toString());
      }
      Text text = new Text();
      for (int i = lines.size() - 1; i >= 0; --i) {
        text.set(lines.get(i));
        context.write(NullWritable.get(), text);
      }
    }
  }
}
