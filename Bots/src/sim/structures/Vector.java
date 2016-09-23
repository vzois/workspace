package sim.structures;

import java.util.Random;
import math.geom2d.Point2D;

public class Vector extends Point2D{
	
	public Vector(double x, double y){
		super(x,y);
	}
	
	public Vector(){
		super(0,0);
	}
	
	public Vector(Vector v){
		super(v.getX(),v.getY());
	}
	
	public Vector(Point2D v){
		super(v.getX(),v.getY());
	}
	
	public Vector(Vector start, Vector end){
		super(end.getX() - start.getX(),end.getY() - start.getY());
	}
	
	public Vector(Vector start, Point2D end){
		super(end.getX() - start.getX(),end.getY() - start.getY());
	}
	
	public void set(double x, double y){
		this.x=x;
		this.y=y;
	}
	
	public void points(Vector start,Vector end){
		this.x = end.x - start.x;
		this.y = end.y - start.y;
	}

	public Double getM(){
		return Math.sqrt(x*x + y*y);
	}
	
	public boolean equals(Vector v){
		if(v.getX() == x && v.getY()==y) return true;
		return false;
	}
	
	public static Vector rand(int minX,int maxX,int minY,int maxY, Random seed){// SET BOT TO RANDOM LOCATION//
		//System.out.println(minX +" "+maxX+" | "+minY+" "+maxX);
		int x=seed.nextInt(maxX - minX+1) + (int)Math.ceil(minX);
		int y=seed.nextInt(maxY - minY+1) + (int)Math.ceil(minY);
		//System.out.println("("+x+","+y+")");
		return new Vector(x,y);		
	}
	
	public void randD(double max, Random seed){
		this.x = seed.nextGaussian()*max;
		this.y = seed.nextGaussian()*max;
	}
	
	public static Vector rand(double max, Random seed){
		return new Vector(seed.nextGaussian()*max,seed.nextGaussian()*max);
	}
	
	public static double det(Vector p, Vector q){ return p.getX()*q.getY() - p.getY()*q.getX(); }
	
	public void normalize(){
		double norm = this.norm();
		if(norm!=0){
			this.x = this.x/norm;
			this.y = this.y/norm;
		}else{
			this.x = 0;
			this.y = 0;
		}
	}
	
	public double norm(){
		return Math.sqrt(this.x*this.x + this.y*this.y);
	}
	
	public double dot(Vector b){
		return this.x*b.getX() + this.y*b.getY();
	}
	
	public void orient(double angle){
		double h = norm();
		this.x = new Double(Math.cos(angle)*h);
		this.y = new Double(Math.sin(angle)*h);
	}
	
	public void multi(double k){
		this.x = this.x*k;
		this.y = this.y*k;
	}
	
	public static Vector add(Vector a, Vector b){
		return new Vector(a.getX()+b.getX(), a.getY() + b.getY());
	}
	
	public static Vector minus(Vector a,Vector b){
		return new Vector(a.getX() - b.getX(),a.getY() - b.getY());
	}
	
	public void add(Vector b){
		this.x = this.x + b.getX();
		this.y = this.y + b.getY();
	}
	
	public void reverse(){
		this.x= -this.x;
		this.y= -this.y;
	}
}
