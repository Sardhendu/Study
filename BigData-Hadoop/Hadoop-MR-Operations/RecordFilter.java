import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
// import org.apache.hadoop.io.LongWrittable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;


public class RecordFilter{
	public static class MapTask extends Mapper<Object, Text, Text, NullWritable>{
		public void map(Object key, Text value, Context context) throws IOException, InterruptedException{
			String token_arr = new String(value.toString());
			String[] token_split = token_arr.split(",");
			int[] fields = new int[token_split.length -1];
			String name = token_split[0];
			
			System.out.print(name);
			
			int i = 0;
			for (String tkn: token_split){
				if (i!=0){
					int field_value = Integer.parseInt(tkn);
					fields[i-1] = field_value;
				}
				i+=1;
			}
			
			if (fields[0] < fields[2] && fields[1] > fields[3]){
				context.write(value, NullWritable.get());
			}
		}
	}

	public static class ReduceTask extends Reducer<Text, NullWritable, Text, NullWritable>{
		private IntWritable result = new IntWritable();
		// Create the reducer method
		public void reduce(Text key, Iterable<NullWritable> values, Context context)throws IOException, InterruptedException{
			context.write(key, NullWritable.get());
		}
	}

	public static void main(String[] args) throws Exception {
		Configuration conf = new Configuration();
		Job job = Job.getInstance(conf, "Filter Record");
		
		job.setJarByClass(RecordFilter.class);
		
		job.setMapperClass(MapTask.class);
		// job.setCombinerClass(ReduceTask.class);
		job.setReducerClass(ReduceTask.class);
		
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(NullWritable.class);
		
		FileInputFormat.addInputPath(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]));
		
		System.exit(job.waitForCompletion(true) ? 0 : 1);
	}
}