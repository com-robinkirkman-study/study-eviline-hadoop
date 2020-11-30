package com.robinkirkman.study.eviline.hadoop;

import java.io.ByteArrayOutputStream;
import java.io.DataOutput;
import java.io.DataOutputStream;
import java.io.IOException;
import java.security.SecureRandom;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Deque;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.TreeMap;

import org.apache.avro.hadoop.io.AvroDatumConverterFactory.AvroWrapperConverter;
import org.apache.avro.mapred.AvroKey;
import org.apache.avro.mapred.AvroValue;
import org.apache.avro.specific.SpecificData;
import org.apache.avro.specific.SpecificRecord;
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
  public static final String MUTATIONS = FitnessJob.class.getName() + ".mutations";

  public static final int GARBAGE_HEIGHT_DEFAULT = 10;
  public static final int LOOKAHEAD_DEFAULT = 3;
  public static final int RETRIES_DEFAULT = 10;
  public static final int GENERATION_SIZE_DEFAULT = 100;
  public static final int MUTATIONS_DEFAULT = 10;
  
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

  private static <D extends SpecificRecord> D deepCopy(D value) {
    return SpecificData.get().deepCopy(value.getSchema(), value);
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

    private int mutations;
    private Deque<FitnessCoefficients> keys;

    public FitnessRecordReader(RecordReader<LongWritable, Text> lines) {
      this.lines = lines;
    }

    @Override
    public void initialize(InputSplit split, TaskAttemptContext context) throws IOException, InterruptedException {
      mutations = context.getConfiguration().getInt(MUTATIONS, MUTATIONS_DEFAULT);
      lines.initialize(split, context);
    }

    @Override
    public boolean nextKeyValue() throws IOException, InterruptedException {
      if (!keys.isEmpty()) {
        keys.poll();
        return true;
      }
      while (lines.nextKeyValue()) {
        String line = lines.getCurrentValue().toString();
        if (line.contains("#"))
          line = line.substring(0, line.indexOf('#'));
        String[] coefficients = line.split(",");
        if (coefficients.length == 1 && coefficients[0].trim().isEmpty())
          continue;
        Random random = new SecureRandom(line.getBytes());
        FitnessCoefficients key = new FitnessCoefficients();
        key.setCoefficients(new ArrayList<>(coefficients.length));
        for (String c : coefficients) {
          key.getCoefficients().add(Double.parseDouble(c));
        }
        keys.offer(key);
        for (int i = 0; i < mutations; ++i) {
          FitnessCoefficients mutation = new FitnessCoefficients();
          mutation.setCoefficients(new ArrayList<>(coefficients.length));
          for (Double d : key.getCoefficients())
            mutation.getCoefficients().add(d + 0.5 - random.nextDouble());
          keys.offer(mutation);
        }
        return true;
      }
      return false;
    }

    @Override
    public FitnessCoefficients getCurrentKey() throws IOException, InterruptedException {
      return keys.peek();
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

  @FunctionalInterface
  public static interface WriteFunction<K, V> {
    void write(K k, V v) throws IOException, InterruptedException;
  }

  public static class FitnessMapping {
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

    public FitnessMapping(int garbageHeight, int lookahead, int retries, int generationSize) {
      this.garbageHeight = garbageHeight;
      this.lookahead = lookahead;
      this.retries = retries;
      this.generationSize = generationSize;
    }

    public void setup() {
      best = new PriorityQueue<>(generationSize, new FitnessComparator());
    }

    public void map(FitnessCoefficients key) {
      Random random = new SecureRandom(Bytes.toArray(key.getCoefficients()));

      FitnessResultCoefficients out = new FitnessResultCoefficients();
      out.setResult(new FitnessResult());
      out.setCoefficients(deepCopy(key));

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

      if (best.size() < generationSize) {
        best.offer(deepCopy(out));
      } else if (best.comparator().compare(out, best.peek()) > 0) {
        best.poll();
        best.offer(deepCopy(out));
      }
    }

    public void cleanup(WriteFunction<NullWritable, FitnessResultCoefficients> context)
        throws IOException, InterruptedException {
      while (!best.isEmpty()) {
        context.write(NullWritable.get(), best.poll());
      }
    }
  }

  public static class FitnessReduction {
    private int generationSize;
    private PriorityQueue<FitnessResultCoefficients> best;

    public FitnessReduction(int generationSize) {
      this.generationSize = generationSize;
    }

    public void setup() {
      best = new PriorityQueue<>(generationSize, new FitnessComparator());
    }

    public void reduce(FitnessResultCoefficients value) {
      if (best.size() < generationSize) {
        best.offer(deepCopy(value));
      } else if (best.comparator().compare(value, best.peek()) > 0) {
        best.poll();
        best.offer(deepCopy(value));
      }
    }

    public void cleanup(WriteFunction<NullWritable, Text> context) throws IOException, InterruptedException {
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

  public static class FitnessMapper
      extends Mapper<FitnessCoefficients, NullWritable, NullWritable, FitnessResultCoefficients> {
    private FitnessMapping mapping;

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
      int garbageHeight = context.getConfiguration().getInt(GARBAGE_HEIGHT, GARBAGE_HEIGHT_DEFAULT);
      int lookahead = context.getConfiguration().getInt(LOOKAHEAD, LOOKAHEAD_DEFAULT);
      int retries = context.getConfiguration().getInt(RETRIES, RETRIES_DEFAULT);
      int generationSize = context.getConfiguration().getInt(GENERATION_SIZE, GENERATION_SIZE_DEFAULT);

      mapping = new FitnessMapping(garbageHeight, lookahead, retries, generationSize);
      mapping.setup();
    }

    @Override
    protected void map(FitnessCoefficients key, NullWritable value, Context context)
        throws IOException, InterruptedException {
      mapping.map(key);
    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
      mapping.cleanup(context::write);
    }
  }

  public static class FitnessReducer extends Reducer<NullWritable, FitnessResultCoefficients, NullWritable, Text> {
    private FitnessReduction reduction;

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
      int generationSize = context.getConfiguration().getInt(GENERATION_SIZE, GENERATION_SIZE_DEFAULT);
      reduction = new FitnessReduction(generationSize);
    }

    @Override
    protected void reduce(NullWritable key, Iterable<FitnessResultCoefficients> values, Context context)
        throws IOException, InterruptedException {
      for (FitnessResultCoefficients value : values) {
        reduction.reduce(value);
      }
    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
      reduction.cleanup(context::write);
    }
  }
}
