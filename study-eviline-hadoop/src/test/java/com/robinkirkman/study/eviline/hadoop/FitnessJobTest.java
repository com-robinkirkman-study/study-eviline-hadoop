package com.robinkirkman.study.eviline.hadoop;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

import org.apache.hadoop.io.Text;
import org.junit.Assert;
import org.junit.Test;

public class FitnessJobTest {
  @Test
  public void reductionLimitsGeneration() throws Exception {
    FitnessJob.FitnessReduction reduction = new FitnessJob.FitnessReduction(10);
    List<Text> reduced = new ArrayList<>();

    FitnessResultCoefficients value = FitnessResultCoefficients.newBuilder().build();
    
    reduction.setup();
    for (int i = 0; i < 5; ++i) {
      for (int j = 0; j < 5; ++j) {
        value.getResult().setRemainingGarbage(i);
        value.getResult().setLinesCleared(j);
        value.getCoefficients().getCoefficients().clear();
        value.getCoefficients().getCoefficients().addAll(Arrays.asList(1d, 2d, 3d, 4d, 5d));
        reduction.reduce(value);
      }
    }
    reduction.cleanup((k, v) -> reduced.add(v));

    Assert.assertEquals(10, reduced.size());
  }
}
