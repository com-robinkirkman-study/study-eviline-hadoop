package com.robinkirkman.study.eviline.hadoop;

import java.io.IOException;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Random;
import java.util.function.Consumer;

import org.apache.curator.shaded.com.google.common.primitives.Bytes;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputFormat;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.OutputCommitter;
import org.apache.hadoop.mapreduce.OutputFormat;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.RecordWriter;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

public class FitnessFormat {
  public static boolean parseCoefficients(String line, FitnessCoefficients parsed) {
    if (line.contains("#"))
      line = line.substring(0, line.indexOf('#'));
    String[] coefficients = line.split(",");
    if (coefficients.length == 1 && coefficients[0].trim().isEmpty())
      return false;
    parsed.setCoefficients(new ArrayList<>(coefficients.length));
    for (String c : coefficients) {
      parsed.getCoefficients().add(Double.parseDouble(c.trim()));
    }
    return true;
  }

  public static void mutateCoefficients(FitnessCoefficients parent, int count, Consumer<FitnessCoefficients> children) {
    Random random = new SecureRandom(Bytes.toArray(parent.getCoefficients()));
    for (int i = 0; i < count; ++i) {
      FitnessCoefficients child = new FitnessCoefficients();
      child.setCoefficients(new ArrayList<>(parent.getCoefficients().size()));
      for (Double d : parent.getCoefficients())
        child.getCoefficients().add(d + 0.5 - random.nextDouble());
      children.accept(child);
    }
  }

  public static String renderCoefficients(FitnessCoefficients out) {
    StringBuilder line = new StringBuilder();
    for (Double d : out.getCoefficients()) {
      if (line.length() > 0)
        line.append(",");
      line.append(String.format("%26f", d.doubleValue()));
    }
    return line.toString();
  }

  public static String renderResultCoefficients(FitnessCoefficientsResult out) {
    StringBuilder line = new StringBuilder();
    line.append(renderCoefficients(out.getCoefficients()));
    line.append(" # remaining_garbage: ");
    line.append(out.getResult().getRemainingGarbage());
    line.append(" lines_cleared: ");
    line.append(out.getResult().getLinesCleared());
    return line.toString();
  }

  public static class FitnessRecordReader extends RecordReader<FitnessCoefficients, NullWritable> {
    private RecordReader<LongWritable, Text> lines;

    private int mutations;;
    private Deque<FitnessCoefficients> keys;

    public FitnessRecordReader(RecordReader<LongWritable, Text> lines, int mutations) {
      this.lines = lines;
      this.mutations = mutations;
    }

    @Override
    public void initialize(InputSplit split, TaskAttemptContext context) throws IOException, InterruptedException {
      lines.initialize(split, context);
    }

    @Override
    public boolean nextKeyValue() throws IOException, InterruptedException {
      while (keys.isEmpty() && lines.nextKeyValue()) {
        String line = lines.getCurrentValue().toString();

        FitnessCoefficients parent = new FitnessCoefficients();
        if (parseCoefficients(line, parent)) {
          keys.offerLast(parent);
          mutateCoefficients(parent, mutations, keys::offerLast);
        }
      }
      return !keys.isEmpty();
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
      int mutations = context.getConfiguration().getInt(FitnessJob.MUTATIONS, FitnessJob.MUTATIONS_DEFAULT);
      return new FitnessRecordReader(text.createRecordReader(split, context), mutations);
    }
  }

  public static class FitnessRecordWriter extends RecordWriter<NullWritable, FitnessCoefficientsResult> {
    private RecordWriter<NullWritable, Text> text;

    public FitnessRecordWriter(RecordWriter<NullWritable, Text> text) {
      this.text = text;
    }

    @Override
    public void write(NullWritable key, FitnessCoefficientsResult value) throws IOException, InterruptedException {
      text.write(key, new Text(renderResultCoefficients(value)));
    }

    @Override
    public void close(TaskAttemptContext context) throws IOException, InterruptedException {
      text.close(context);
    }
  }

  public static class FitnessOutputFormat extends OutputFormat<NullWritable, FitnessCoefficientsResult> {
    private TextOutputFormat<NullWritable, Text> text = new TextOutputFormat<>();

    @Override
    public RecordWriter<NullWritable, FitnessCoefficientsResult> getRecordWriter(TaskAttemptContext context)
        throws IOException, InterruptedException {
      // TODO Auto-generated method stub
      return null;
    }

    @Override
    public void checkOutputSpecs(JobContext context) throws IOException, InterruptedException {
      text.checkOutputSpecs(context);
    }

    @Override
    public OutputCommitter getOutputCommitter(TaskAttemptContext context) throws IOException, InterruptedException {
      return text.getOutputCommitter(context);
    }

  }

  private FitnessFormat() {
  }
}
