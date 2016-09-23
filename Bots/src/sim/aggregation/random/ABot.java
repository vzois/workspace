package sim.aggregation.random;
import java.awt.Color;

import sim.elements.Bot;
import sim.elements.Thing;


public class ABot extends Bot{
	boolean prey = false;
	private int time;
	private int bias;
	Thing t=null;
	Color c1 = Color.CYAN;
	Color c2 = Color.BLUE;
	
	public ABot(double size,double range,int bias){
		super(size,range,Color.BLUE);
		this.bias=bias;
		this.setMaxSpeed(5);
		time=0;
	}
	
	public void act(){
		
		if(!prey){
			Object o = null;
			for(int i = 0;i<this.getNeighbors().length;i++){
				o = this.getNeighbors()[i];
				if(o instanceof Prey || ( (o instanceof Bot) && ((Bot)o).getC()==Color.RED) && !o.equals(this)){
					prey=true;
					t = (Thing)o;
					break;
				}
			}
		}
		
		if(prey){
			if(this.getCenter().distance(t.getCenter())>=size+t.getS()){
				velocity.points(this.getCenter(),t.getCenter());
				this.limitSpeed();
				color = Color.YELLOW;
			}else{
				this.stop();
				color = Color.RED;
			}
		}else if(time%bias==0){
			this.velocity.randD(this.getMaxSpeed(), seed);
		}	
		time++;
	}
}
