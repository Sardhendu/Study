package com.hadoop_partitionar;

import java.io.IOException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class HP_driver
{
  public static void main(String[] args)
    throws IOException, InterruptedException, ClassNotFoundException
  {
    Job job_run = new Job(new Configuration());
    
    job_run.setJobName("sort_by_partition");
    
    job_run.setJarByClass(HP_driver.class);
    
    job_run.setMapperClass(HP_mapper.class);
    job_run.setCombinerClass(HP_reducer.class);
    job_run.setPartitionerClass(HP_partitionar.class);
    job_run.setReducerClass(HP_reducer.class);
    
    job_run.setMapOutputKeyClass(IntWritable.class);
    job_run.setMapOutputValueClass(IntWritable.class);
    job_run.setOutputKeyClass(IntWritable.class);
    job_run.setOutputValueClass(IntWritable.class);
    job_run.setNumReduceTasks(2);
    
    FileInputFormat.setInputPaths(job_run, new Path[] { new Path("/home/hduser/sort_input.txt") });
    FileOutputFormat.setOutputPath(job_run, new Path("/home/hduser/sort_output"));
    
    job_run.waitForCompletion(true);
  }
}
