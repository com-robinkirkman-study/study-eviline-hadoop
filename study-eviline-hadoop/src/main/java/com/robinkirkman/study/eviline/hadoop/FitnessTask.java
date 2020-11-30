package com.robinkirkman.study.eviline.hadoop;

import java.io.IOException;
import java.security.SecureRandom;
import java.util.ArrayDeque;
import java.util.Collections;
import java.util.Comparator;
import java.util.Deque;
import java.util.Iterator;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.function.Function;

import org.apache.avro.specific.SpecificData;
import org.apache.avro.specific.SpecificRecord;
import org.apache.curator.shaded.com.google.common.primitives.Bytes;
import org.eviline.core.Block;
import org.eviline.core.Engine;
import org.eviline.core.Field;
import org.eviline.core.ai.AIPlayer;
import org.eviline.core.ai.DefaultAIKernel;
import org.eviline.core.ai.DefaultFitness;
import org.eviline.core.ai.Player;

public class FitnessTask {
  @FunctionalInterface
  public static interface FitnessWriter {
    void write(FitnessCoefficientsResult value) throws IOException, InterruptedException;
  }

  public static class FitnessCoefficientsComparator implements Comparator<FitnessCoefficients> {
    public int compare(FitnessCoefficients lhs, FitnessCoefficients rhs) {
      Iterator<Double> li = lhs.getCoefficients().iterator();
      Iterator<Double> ri = rhs.getCoefficients().iterator();
      while (li.hasNext() && ri.hasNext()) {
        int c = Double.compare(li.next(), ri.next());
        if (c != 0)
          return c;
      }
      if (li.hasNext())
        return 1;
      if (ri.hasNext())
        return -1;
      return 0;
    }
  }

  public static class FitnessResultComparator implements Comparator<FitnessResult> {
    @Override
    public int compare(FitnessResult lhs, FitnessResult rhs) {
      if (lhs == null && rhs == null)
        return 0;
      else if (lhs == null)
        return -1;
      else if (rhs == null)
        return 1;

      int c = -Long.compare(lhs.getRemainingGarbage(), rhs.getRemainingGarbage());
      if (c != 0)
        return c;
      return -Long.compare(lhs.getLinesCleared(), rhs.getLinesCleared());
    }
  }

  public static Comparator<FitnessResult> FITNESS_RESULT_ORDER = new FitnessResultComparator();
  public static Comparator<FitnessCoefficients> FITNESS_COEFFICIENTS_ORDER = new FitnessCoefficientsComparator();

  private static <D extends SpecificRecord> D deepCopy(D value) {
    return SpecificData.get().deepCopy(value.getSchema(), value);
  }

  public static class FitnessMapper {
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

    public static Function<FitnessCoefficients, FitnessResult> newDefaultEvaluator(int garbageHeight, int lookahead,
        int retries) {
      return (FitnessCoefficients key) -> {
        Random random = new SecureRandom(Bytes.toArray(key.getCoefficients()));
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
          DefaultFitness fitness = new DefaultFitness();
          DefaultAIKernel aiKernel = new DefaultAIKernel(fitness);
          Player player = new AIPlayer(aiKernel, engine, lookahead);

          while (!engine.isOver()) {
            engine.tick(player.tick());
            if (countGarbageLines(field) == 0)
              break;
          }

          linesCleared += engine.getLines();
          remainingGarbage += countGarbageLines(field);
        }

        FitnessResult value = new FitnessResult();
        value.setLinesCleared(linesCleared);
        value.setRemainingGarbage(remainingGarbage);
        return value;
      };
    }

    private int generationSize;

    private Function<FitnessCoefficients, FitnessResult> evaluator;

    private PriorityQueue<FitnessCoefficientsResult> best;

    public FitnessMapper(int garbageHeight, int lookahead, int retries, int generationSize) {
      this(newDefaultEvaluator(garbageHeight, lookahead, retries), generationSize);
    }

    public FitnessMapper(Function<FitnessCoefficients, FitnessResult> evaluator, int generationSize) {
      this.evaluator = evaluator;
      this.generationSize = generationSize;
    }

    public void setup() {
      best = new PriorityQueue<>(generationSize,
          (lhs, rhs) -> FITNESS_RESULT_ORDER.compare(lhs.getResult(), rhs.getResult()));
    }

    public void map(FitnessCoefficients key) {
      FitnessCoefficientsResult result = new FitnessCoefficientsResult();
      result.setCoefficients(deepCopy(key));
      result.setResult(evaluator.apply(result.getCoefficients()));

      if (best.size() < generationSize) {
        best.offer(deepCopy(result));
      } else if (best.comparator().compare(result, best.peek()) > 0) {
        best.poll();
        best.offer(deepCopy(result));
      }
    }

    public void cleanup(FitnessWriter writer) throws IOException, InterruptedException {
      while (!best.isEmpty()) {
        writer.write(best.poll());
      }
    }
  }

  public static class FitnessReducer {
    private int generationSize;
    private PriorityQueue<FitnessCoefficientsResult> best;

    public FitnessReducer(int generationSize) {
      this.generationSize = generationSize;
    }

    public void setup() {
      best = new PriorityQueue<>(generationSize,
          (lhs, rhs) -> FITNESS_RESULT_ORDER.compare(lhs.getResult(), rhs.getResult()));
    }

    public void reduce(FitnessCoefficientsResult value) {
      if (best.size() < generationSize) {
        best.offer(deepCopy(value));
      } else if (best.comparator().compare(value, best.peek()) > 0) {
        best.poll();
        best.offer(deepCopy(value));
      }
    }

    public void cleanup(FitnessWriter writer) throws IOException, InterruptedException {
      Deque<FitnessCoefficientsResult> buf = new ArrayDeque<>(best.size());
      while (!best.isEmpty()) {
        buf.offerLast(best.poll());
      }
      while (!buf.isEmpty()) {
        writer.write(buf.pollLast());
      }
    }
  }
}
