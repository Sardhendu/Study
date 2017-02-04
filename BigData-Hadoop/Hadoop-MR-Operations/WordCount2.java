// The module uses MapReduce API


// Input:
// Sam is my name. My name was Sam. Sardhendu Mishra is my made-up name. Call me Sam or Sardhendu.
// Ouptut:
// Other	10
// Small	9


import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
// import org.apache.hadoop.io.LongWrittable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;


public class WordCount2{
	public static class MapTask extends Mapper<Object, Text, Text, IntWritable>{
		private static final IntWritable one = new IntWritable(1);
		private Text token_a_n = new Text("Small");
		private Text token_other = new Text("Other");
		// Create the map method
		public void map(Object key, Text value, Context context) throws IOException, InterruptedException{
			// Tokenize the first line into tokens
			StringTokenizer token_arr = new StringTokenizer(value.toString());
			// Iterate through each tokens
			while (token_arr.hasMoreTokens()){
				String token = token_arr.nextToken();
				// Check condition here (Using Regex to get if the 1st character matches)
				// if (token.startsWith("a-n")){
				if (token.matches("^[a-n].*$")){
					context.write(token_a_n, one);
				}
				else{
					context.write(token_other, one);
				}
				
			}

		}
	}

	public static class ReduceTask extends Reducer<Text, IntWritable, Text, IntWritable>{
		private IntWritable result = new IntWritable();
		// Create the reducer method
		public void reduce(Text key, Iterable<IntWritable> values, Context context)throws IOException, InterruptedException{
			int sum = 0;
			for (IntWritable val : values) {
				sum += val.get();
			}
			result.set(sum);
			context.write(key, result);
		}
	}

	public static void main(String[] args) throws Exception {
		Configuration conf = new Configuration();
		Job job = Job.getInstance(conf, "word count 2");
		
		job.setJarByClass(WordCount2.class);
		
		job.setMapperClass(MapTask.class);
		// job.setCombinerClass(ReduceTask.class);
		job.setReducerClass(ReduceTask.class);
		
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(IntWritable.class);
		
		FileInputFormat.addInputPath(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]));
		
		System.exit(job.waitForCompletion(true) ? 0 : 1);
	}
}




