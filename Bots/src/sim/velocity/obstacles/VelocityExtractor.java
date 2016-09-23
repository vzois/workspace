package sim.velocity.obstacles;

import java.util.ArrayList;
import java.util.Random;
import sim.structures.Vector;

public class VelocityExtractor {
	protected Vector location;
	protected Vector preferredVelocity;
	protected double obstacleSize;
	
	public VelocityExtractor(Vector location,Vector preferredVelocity,double obstacleSize){
		this.location = new Vector(location);
		this.preferredVelocity = new Vector(preferredVelocity);
		this.obstacleSize = obstacleSize;
	}
	
	public ArrayList<Vector> extractAngularVelocities(int step,int max){
		ArrayList<Vector> velocities = new ArrayList<Vector>();
		//velocities.add(this.preferredVelocity);
		Vector vi;
		
		for(int i=-max;i<max;i+=step){
			vi = new Vector(this.preferredVelocity);
			vi.orient(i);
			vi.multi(obstacleSize);
			velocities.add(vi);
		}
		return velocities;
	}
	
	public ArrayList<Vector> extractGridVelocities(int min,int max){
		ArrayList<Vector> velocities = new ArrayList<Vector>((max-min)*(max-min));
		//velocities.add(preferredVelocity);
		Vector vi = null;
		
		for(int i=min;i<=max;i++){
			for(int j=min;j<=max;j++){
				vi = new Vector(i,j);
				vi.normalize();
				vi.multi(obstacleSize);
				velocities.add(new Vector(i,j));
			}
		}
		return velocities;
	}
	
	public ArrayList<Vector> extractRandom(int num,int max,Random seed){
		ArrayList<Vector> velocities = new ArrayList<Vector>(num);
		//velocities.add(preferredVelocity);
		for(int i=0;i<num;i++){
			velocities.add(Vector.rand(max, seed));
		}
		return velocities;
	}
}
