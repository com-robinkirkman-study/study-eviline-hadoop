package com.robinkirkman.study.eviline.hadoop;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.junit.Assert;
import org.junit.Test;

public class FitnessTaskTest {
  @Test
  public void reductionLimitsGeneration() throws Exception {
    FitnessTask.FitnessReducer reducer = new FitnessTask.FitnessReducer(10);
    List<FitnessCoefficientsResult> reduced = new ArrayList<>();

    FitnessCoefficientsResult value = FitnessCoefficientsResult.newBuilder().build();

    reducer.setup();
    for (int i = 0; i < 5; ++i) {
      for (int j = 0; j < 5; ++j) {
        value.getResult().setRemainingGarbage(i);
        value.getResult().setLinesCleared(j);
        value.getCoefficients().getCoefficients().clear();
        value.getCoefficients().getCoefficients().addAll(Arrays.asList(1d, 2d, 3d, 4d, 5d));
        reducer.reduce(value);
      }
    }
    reducer.cleanup(reduced::add);

    Assert.assertEquals(10, reduced.size());
  }
}
