package sim.collision.avoidance.swarm;

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
public class SwarmCollisionAvoidanceWindow extends Window{
	int x=800,y=800;
	int pMaxAgentNumber=100,pMinAgentNumber=1,pAgentNumber=10;
	
	JSlider sliderP;
	JLabel labelP;
	JCheckBox cb;
	
	@Override
	public World createWorld() {
		World world = new World(x,y);
		for(int i=0;i<pAgentNumber;i++){
			world.addObject(new ColorFullBot(new Color(world.seed.nextFloat(),world.seed.nextFloat(),world.seed.nextFloat())));
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
		labelP = new JLabel("Number of Agents: "+this.pAgentNumber);
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
}
