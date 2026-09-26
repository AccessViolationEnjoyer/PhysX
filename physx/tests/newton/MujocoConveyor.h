#ifndef NEWTON_MUJOCO_CONVEYOR_H
#define NEWTON_MUJOCO_CONVEYOR_H

// Contact adjustments shared by the MuJoCo conveyor runners. Belts are the first geoms of
// each scene and move along +x; MuJoCo has no contact target-velocity setter.
#include <mujoco/mujoco.h>

namespace mujocoConveyor
{
// MuJoCo's solimp impedance (a smooth step from dmin to dmax over width) at a violation.
inline double impedance(const mjtNum* solimp, double violation)
{
	const double dmin = mju_clip(solimp[0], mjMINIMP, mjMAXIMP), dmax = mju_clip(solimp[1], mjMINIMP, mjMAXIMP);
	const double width = solimp[2], midpoint = solimp[3], power = solimp[4];
	if(dmin == dmax || width <= mjMINVAL)
		return 0.5 * (dmin + dmax);
	const double x = mju_min(1.0, mju_abs(violation) / width);
	const double y = x <= midpoint ? mju_pow(x, power) / mju_pow(midpoint, power - 1.0) :
		1.0 - mju_pow(1.0 - x, power) / mju_pow(1.0 - midpoint, power - 1.0);
	return dmin + y * (dmax - dmin);
}

// Keep MuJoCo's positive contact-detection margin separate from rest distance.
// As in PhysX, the intended resting surfaces touch; the collision envelope
// must not create a visible gap between resting bodies.
// MuJoCo measures both the spring and the impedance from the margin. Measure both
// from contact instead, so solimp stiffens with penetration as PhysX Newton's
// surface regularization does. Constant solimp only removes the margin offset.
// The conveyor scenes have contact rows only.
inline void applyRestDistance(mjData* data)
{
	for(int i = 0; i < data->ncon; ++i)
	{
		const mjContact& contact = data->contact[i];
		if(contact.efc_address < 0)
			continue;
		const int rows = contact.dim == 1 ? 1 : 2 * (contact.dim - 1);
		for(int row = contact.efc_address; row < contact.efc_address + rows; ++row)
		{
			mjtNum* kbip = data->efc_KBIP + 4 * row;
			const double position = data->efc_pos[row];
			const double previous = kbip[2];
			const double current = impedance(contact.solimp, mju_min(0.0, position));
			data->efc_aref[row] += kbip[0] * (previous * (position - data->efc_margin[row]) - current * position);
			data->efc_R[row] *= (1.0 - current) * previous / ((1.0 - previous) * current);
			data->efc_D[row] = 1.0 / data->efc_R[row];
			kbip[2] = current;
		}
	}
	// mj_step1 already copied R and D into the island arrays; aref is copied later.
	if(data->nisland)
	{
		for(int islandRow = 0; islandRow < data->nefc; ++islandRow)
		{
			const int row = data->map_iefc2efc[islandRow];
			data->iefc_R[islandRow] = data->efc_R[row];
			data->iefc_D[islandRow] = data->efc_D[row];
		}
	}
}

// Supply the belts' prescribed surface velocity in the acceleration reference, before
// the constraint solve. Pyramidal rows are normal +/- mu*tangent, in geom[1]-geom[0] order.
inline void applyConveyorVelocity(mjData* data, int beltCount, double beltSpeed)
{
	for(int i = 0; i < data->ncon; ++i)
	{
		const mjContact& contact = data->contact[i];
		const bool beltFirst = contact.geom[0] < beltCount;
		const bool beltSecond = contact.geom[1] < beltCount;
		if(contact.efc_address < 0 || (!beltFirst && !beltSecond))
			continue;
		const double speed = beltFirst ? beltSpeed : -beltSpeed;
		for(int tangent = 0; tangent < 2; ++tangent)
		{
			for(int sign = 0; sign < 2; ++sign)
			{
				const int row = contact.efc_address + tangent * 2 + sign;
				const double target = speed * (contact.frame[0] + (sign ? -1.0 : 1.0) *
					contact.friction[tangent] * contact.frame[(tangent + 1) * 3]);
				data->efc_vel[row] -= target;
				data->efc_aref[row] += data->efc_KBIP[4 * row + 1] * target;
			}
		}
	}
}
}
#endif
