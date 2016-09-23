package sim.collision.avoidance.lineofrobots;

import java.awt.Color;

import javax.swing.Box;
import javax.swing.JCheckBox;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import sim.graphics.Window;
import sim.graphics.World;

@SuppressWarnings("serial")
public class LineCollisionWindow extends Window {
	int x=800,y=800;
	int pMaxAgentNumber=10,pMinAgentNumber=1,pAgentNumber=1;
	int yMaxAgentNumber=10,yMinAgentNumber=1,yAgentNumber=1;
	
	JSlider sliderP,sliderY;
	JLabel labelP,labelY;
	JCheckBox cb;
	
	@Override
	public World createWorld() {
		World world = new World(x,y);
		TransparentBot tb;
		for(int i=0;i<this.pAgentNumber;i++){
			tb = new TransparentBot(Color.MAGENTA);
			world.addObject(tb,0,100,0,100);
			tb.setL(400, 750 - i*45);
		}
		
		for(int i=0;i<this.yAgentNumber;i++){
			tb = new TransparentBot(Color.YELLOW);
			world.addObject(tb,0,100,0,100);
			tb.setL(400, 50 + i*45);
		}
		return world;
	}

	@Override
	public JPanel getSettings() {
		JPanel panel = new JPanel();
		Box settingsBox = Box.createVerticalBox();
		settingsBox.setAlignmentX(CENTER_ALIGNMENT);
		settingsBox.setAlignmentY(CENTER_ALIGNMENT);
		
		cb = new JCheckBox("Draw Velocity Obstacles",false);
		cb.addChangeListener(new ChangeListener(){
			@Override
			public void stateChanged(ChangeEvent e) {
				changeDrawVelocityObstacles(e);
			}
			
		});
		cb.setEnabled(true);
		cb.setSelected(false);
		settingsBox.add(cb);
		///////////////////////////////////////////////////////////////////
		sliderP = new JSlider();
		labelP = new JLabel("Purple Agent Number: "+this.pAgentNumber);
		settingsBox.add(labelP);
		sliderP.setMinimum(this.pMinAgentNumber);
		sliderP.setMaximum(this.pMaxAgentNumber);
		sliderP.setValue(this.pAgentNumber);
		sliderP.addChangeListener(new ChangeListener()
		{
			public void stateChanged(final ChangeEvent event)
			{
				agentsChangeP(event);
			}
		});		
		settingsBox.add(sliderP);
		///////////////////////////////////////////////////////////////////
		sliderY = new JSlider();
		labelY = new JLabel("Yellow Agent Number: "+this.yAgentNumber);
		settingsBox.add(labelY);
		sliderY.setMinimum(this.yMinAgentNumber);
		sliderY.setMaximum(this.yMaxAgentNumber);
		sliderY.setValue(this.yAgentNumber);
		sliderY.addChangeListener(new ChangeListener()
		{
			public void stateChanged(final ChangeEvent event)
			{
				agentsChangeY(event);
			}
		});		
		settingsBox.add(sliderY);
		
		panel.add(settingsBox);
		return panel;
	}
	
	public void changeDrawVelocityObstacles(ChangeEvent e){
		if(cb.isSelected()){
			World.drawVelocityObstacle = true;
		}else{
			World.drawVelocityObstacle = false;
		}
	}
	
	public void agentsChangeP(ChangeEvent event){
		this.pAgentNumber=sliderP.getValue();
		labelP.setText("Purple Agent Number: "+this.pAgentNumber);
	}
	
	public void agentsChangeY(ChangeEvent event){
		this.yAgentNumber=sliderY.getValue();
		labelY.setText("Yellow Agent Number: "+this.yAgentNumber);
	}

}

